"""Live trading executor for RL-Crypto."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from dotenv import load_dotenv
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from src.data import DataProcessor
from src.live.risk_manager import RiskManager, RiskConfig
from src.live.websocket_client import BinanceWebSocket
from src.live.alerts import AlertManager, Alert, AlertLevel
from src.data.binance_client import BinanceClient

# Load .env file
load_dotenv()


class LiveExecutor:
    """Live trading executor using trained RL model.
    
    Connects to Binance via WebSocket for real-time data,
    runs predictions through trained model, and executes trades.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config/config.yaml",
        risk_config: Optional[RiskConfig] = None,
        dry_run: bool = True,
    ):
        """Initialize live executor.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            risk_config: Risk management configuration
            dry_run: If True, don't execute actual trades
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = model_path
        self.dry_run = dry_run
        self.symbols = self.config["symbols"]
        
        # Initialize components
        risk_cfg = risk_config or RiskConfig(
            max_order_value=2500.0,  # Allow $2500 per order (50% of capital)
            max_position_per_asset=0.5,  # Max 50% per asset
            max_total_exposure=1.0,  # Max 100% exposure
        )
        self.risk_manager = RiskManager(
            config=risk_cfg,
            initial_capital=self.config["trading"]["initial_capital"],
        )
        
        self.data_processor = DataProcessor(
            lookback=self.config["data"]["lookback"]
        )
        
        # Load processor params if available
        params_path = Path(self.config["data"]["data_dir"]) / "processor_params.pkl"
        if params_path.exists():
            self.data_processor.load_params(str(params_path))
        
        # Initialize Binance client for order execution
        self.binance_client = BinanceClient(
            testnet=self.config["binance"].get("testnet", True)
        )
        
        # Load model
        self._load_model()
        
        # Data buffers
        self.kline_buffer: dict[str, list] = {s: [] for s in self.symbols}
        self.latest_features: dict[str, np.ndarray] = {}
        
        # State
        self.is_running = False
        self.ws_client: Optional[BinanceWebSocket] = None
        self.last_actions: dict[str, float] = {s: 0.0 for s in self.symbols}  # For action smoothing
        
        # Alert manager for notifications
        self.alert_manager = AlertManager(
            enable_telegram=True,
            enable_slack=False,
            enable_discord=False,
            rate_limit_seconds=0,  # No rate limiting for trades
        )
        
        logger.info(f"LiveExecutor initialized (dry_run={dry_run})")

    def _load_model(self) -> None:
        """Load trained PPO model."""
        self.model = PPO.load(self.model_path)
        
        # VecNormalize is not needed for inference without env
        self.vec_normalize_stats = None
        
        logger.info(f"Model loaded from {self.model_path}")

    async def start(self) -> None:
        """Start live trading."""
        logger.info("=" * 60)
        logger.info("Starting Live Trading")
        logger.info("=" * 60)
        
        if self.dry_run:
            logger.warning("DRY RUN MODE - No actual trades will be executed")
        
        # Pre-load historical data for immediate trading
        await self._load_historical_data()
        
        # Initialize WebSocket client
        self.ws_client = BinanceWebSocket(
            symbols=self.symbols,
            testnet=self.config["binance"].get("testnet", True),
            on_kline=self._on_kline,
        )
        
        await self.ws_client.connect()
        self.is_running = True
        
        # Trigger initial prediction if we have enough data
        if all(len(self.kline_buffer[s]) >= self.config["data"]["lookback"] + 60 for s in self.symbols):
            logger.info("Historical data loaded, starting predictions immediately")
            await self._predict_and_execute()
        
        # Main loop
        try:
            while self.is_running:
                await asyncio.sleep(1)
                
                # Log status periodically
                if datetime.now().second == 0:
                    self._log_status()
                    
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.stop()
    
    async def _load_historical_data(self) -> None:
        """Pre-load historical kline data for immediate trading."""
        import pandas as pd
        
        lookback = self.config["data"]["lookback"]
        timeframe = self.config["data"]["timeframe"]
        
        # Calculate days needed (lookback + buffer)
        bars_needed = lookback + 100
        if timeframe == "1h":
            days_needed = max(1, bars_needed // 24 + 1)
        elif timeframe == "15m":
            days_needed = max(1, bars_needed // 96 + 1)
        else:
            days_needed = max(1, bars_needed // 1440 + 1)
        
        logger.info(f"Loading historical {timeframe} data ({days_needed} days, ~{bars_needed} bars)...")
        
        for symbol in self.symbols:
            try:
                # Fetch historical klines using days parameter
                df = self.binance_client.get_historical_klines(
                    symbol=symbol,
                    interval=timeframe,
                    days=days_needed,
                )
                
                if not df.empty:
                    # Convert DataFrame to buffer format
                    for _, row in df.tail(bars_needed).iterrows():
                        self.kline_buffer[symbol].append({
                            "symbol": symbol,
                            "interval": timeframe,
                            "open_time": row["open_time"],
                            "close_time": row["close_time"],
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"]),
                            "is_closed": True,
                        })
                    
                    logger.info(f"{symbol}: Loaded {len(self.kline_buffer[symbol])} historical bars")
                    
                    # Update features
                    await self._update_features(symbol)
                    
            except Exception as e:
                logger.error(f"Failed to load historical data for {symbol}: {e}")
                import traceback
                traceback.print_exc()

    async def stop(self) -> None:
        """Stop live trading."""
        self.is_running = False
        
        if self.ws_client:
            await self.ws_client.disconnect()
        
        # Close all positions if not dry run
        if not self.dry_run:
            await self._close_all_positions()
        
        logger.info("Live trading stopped")

    async def _on_kline(self, kline: dict) -> None:
        """Handle incoming kline data.
        
        Args:
            kline: Kline data from WebSocket
        """
        symbol = kline["symbol"]
        
        # Only process closed candles
        if not kline["is_closed"]:
            return
        
        # Add to buffer
        self.kline_buffer[symbol].append(kline)
        
        # Keep only lookback + some buffer
        max_buffer = self.config["data"]["lookback"] + 100
        if len(self.kline_buffer[symbol]) > max_buffer:
            self.kline_buffer[symbol] = self.kline_buffer[symbol][-max_buffer:]
        
        # Update features
        await self._update_features(symbol)
        
        # Check if we have enough data for all symbols
        if all(
            len(self.kline_buffer[s]) >= self.config["data"]["lookback"] + 60
            for s in self.symbols
        ):
            # Get prediction and execute
            await self._predict_and_execute()

    async def _update_features(self, symbol: str) -> None:
        """Update features for a symbol.
        
        Args:
            symbol: Symbol to update
        """
        import pandas as pd
        
        klines = self.kline_buffer[symbol]
        if len(klines) < 100:  # Need minimum for indicators
            return
        
        # Convert to DataFrame, extracting only OHLCV columns
        df = pd.DataFrame([
            {
                "open_time": k["open_time"],
                "open": k["open"],
                "high": k["high"],
                "low": k["low"],
                "close": k["close"],
                "volume": k["volume"],
            }
            for k in klines
        ])
        
        try:
            # Process features
            processed = self.data_processor.process(df, fit=False)
            
            if len(processed) >= self.config["data"]["lookback"]:
                # Create sequence (last lookback window)
                features = processed[self.data_processor.feature_columns].values
                self.latest_features[symbol] = features[-self.config["data"]["lookback"]:]
                logger.debug(f"{symbol}: Updated features, shape={self.latest_features[symbol].shape}")
                
        except Exception as e:
            logger.error(f"Error processing features for {symbol}: {e}")

    async def _predict_and_execute(self) -> None:
        """Get model prediction and execute trades."""
        # Check if we have features for all symbols
        if len(self.latest_features) != len(self.symbols):
            return
        
        # Build observation
        market_obs = np.stack([
            self.latest_features[s] for s in self.symbols
        ]).astype(np.float32)
        
        # Get current positions
        positions = np.zeros(len(self.symbols), dtype=np.float32)
        for i, symbol in enumerate(self.symbols):
            if symbol in self.risk_manager.positions:
                positions[i] = self.risk_manager.positions[symbol].size
        
        # Portfolio state
        cash_pct = self.risk_manager.current_capital / self.risk_manager.initial_capital
        value_pct = cash_pct  # Simplified
        
        portfolio_obs = np.concatenate([
            positions,
            np.array([cash_pct, value_pct], dtype=np.float32),
        ])
        
        obs = {
            "market": market_obs,
            "portfolio": portfolio_obs,
        }
        
        # Get prediction
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Execute trades for each symbol
        for i, symbol in enumerate(self.symbols):
            raw_action = float(action[i])
            
            # 1. Action Smoothing (Alpha = 0.8)
            # This reduces signal volatility from the RL model
            last_action = self.last_actions.get(symbol, 0.0)
            alpha = 0.8
            smoothed_action = alpha * last_action + (1 - alpha) * raw_action
            self.last_actions[symbol] = smoothed_action
            
            target_position = smoothed_action
            
            current_price = self.ws_client.get_price(symbol)
            if current_price is None:
                continue
                
            # 2. Deadband (Threshold = 0.05)
            # Only trade if the target position change is significant
            # Note: positions[i] here comes from risk_manager.positions which tracks 'size' (weight)
            current_weight = positions[i]
            if abs(target_position - current_weight) < 0.05:
                # Change too small, skip
                continue
            
            # Apply risk management
            target_position = self.risk_manager.adjust_position_for_risk(
                symbol, target_position
            )
            
            # Check if trade is allowed
            allowed, reason = self.risk_manager.check_trade_allowed(
                symbol, target_position, current_price
            )
            
            if not allowed:
                logger.debug(f"{symbol}: Trade blocked - {reason}")
                continue
            
            # Check stop loss
            should_stop, _ = self.risk_manager.check_stop_loss(symbol)
            if should_stop:
                target_position = 0.0
            
            # Calculate position change
            current_position = positions[i]
            position_change = target_position - current_position
            
            if abs(position_change) < 0.01:
                continue
            
            # Execute trade
            await self._execute_trade(symbol, position_change, current_price)

    async def _execute_trade(
        self,
        symbol: str,
        position_change: float,
        price: float,
    ) -> None:
        """Execute a trade.
        
        Args:
            symbol: Trading symbol
            position_change: Change in position
            price: Execution price
        """
        side = "BUY" if position_change > 0 else "SELL"
        quantity = abs(position_change) * self.risk_manager.current_capital / price
        
        logger.info(
            f"{'[DRY RUN] ' if self.dry_run else ''}"
            f"Trade: {side} {quantity:.4f} {symbol} @ {price:.2f}"
        )
        
        if not self.dry_run:
            try:
                # Execute market order
                order = self.binance_client.place_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                )
                
                if order:
                    logger.info(f"Order executed: {order.get('orderId')}")
                    
                    # Send Telegram notification
                    await self.alert_manager.trade_opened(
                        symbol=symbol,
                        side=side,
                        size=quantity,
                        price=price,
                    )
                    
                    # Update risk manager
                    self.risk_manager.record_trade(0)  # PnL calculated later
                    
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
                await self.alert_manager.error("Order Failed", f"{symbol}: {e}")
        else:
            # DRY RUN notification
            await self.alert_manager.send(Alert(
                AlertLevel.INFO,
                f"[DRY RUN] Trade: {symbol}",
                f"Side: {side}\nQuantity: {quantity:.4f}\nPrice: ${price:.2f}"
            ))
        
        # Update position tracking
        current_pos = self.risk_manager.positions.get(symbol)
        new_size = (current_pos.size if current_pos else 0) + position_change
        
        self.risk_manager.update_position(
            symbol=symbol,
            size=new_size,
            entry_price=price,
            current_price=price,
        )

    async def _close_all_positions(self) -> None:
        """Close all open positions."""
        logger.info("Closing all positions...")
        
        for symbol, pos in list(self.risk_manager.positions.items()):
            if abs(pos.size) > 0.01:
                current_price = self.ws_client.get_price(symbol) if self.ws_client else pos.current_price
                await self._execute_trade(symbol, -pos.size, current_price)

    def _log_status(self) -> None:
        """Log current trading status."""
        metrics = self.risk_manager.get_metrics()
        
        logger.info(
            f"Status: Capital=${metrics['current_capital']:.2f} | "
            f"DD={metrics['current_drawdown']*100:.1f}% | "
            f"Trades={metrics['total_trades']} | "
            f"Positions={metrics['active_positions']}"
        )


async def main():
    """Main entry point for live trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run live trading")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config path")
    parser.add_argument("--live", action="store_true", help="Execute real trades (default: dry run)")
    args = parser.parse_args()
    
    executor = LiveExecutor(
        model_path=args.model,
        config_path=args.config,
        dry_run=not args.live,
    )
    
    await executor.start()


if __name__ == "__main__":
    asyncio.run(main())
