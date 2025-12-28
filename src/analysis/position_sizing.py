"""Position sizing algorithms: Kelly Criterion and Risk Parity."""

from typing import Optional
import numpy as np
from loguru import logger


class KellyCriterion:
    """Kelly Criterion position sizing for optimal growth.
    
    f* = (bp - q) / b = (p * b - (1 - p)) / b
    
    where:
    - f* = optimal fraction of capital
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = win/loss ratio (average win / average loss)
    """
    
    def __init__(
        self,
        lookback: int = 100,
        fraction: float = 0.5,  # "Half Kelly" for safety
        min_trades: int = 30,
        max_fraction: float = 0.25,
    ):
        """Initialize Kelly Criterion calculator.
        
        Args:
            lookback: Number of recent trades to consider
            fraction: Kelly fraction (0.5 = half Kelly for less risk)
            min_trades: Minimum trades required for calculation
            max_fraction: Maximum allowed position fraction
        """
        self.lookback = lookback
        self.fraction = fraction
        self.min_trades = min_trades
        self.max_fraction = max_fraction
        
        # Trade history
        self.trade_results: list[float] = []

    def record_trade(self, pnl_pct: float) -> None:
        """Record a trade result.
        
        Args:
            pnl_pct: Trade PnL as percentage (e.g., 0.02 for 2%)
        """
        self.trade_results.append(pnl_pct)
        
        # Keep only lookback trades
        self.trade_results = self.trade_results[-self.lookback:]

    def calculate_kelly(self) -> tuple[float, dict]:
        """Calculate Kelly fraction based on trade history.
        
        Returns:
            Tuple of (kelly_fraction, stats_dict)
        """
        if len(self.trade_results) < self.min_trades:
            return 0.1, {"error": "Not enough trades"}
        
        trades = np.array(self.trade_results[-self.lookback:])
        
        # Calculate components
        wins = trades[trades > 0]
        losses = trades[trades <= 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.1, {"error": "Need both wins and losses"}
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins)
        avg_loss = np.abs(np.mean(losses))
        
        # Win/loss ratio
        win_loss_ratio = avg_win / (avg_loss + 1e-8)
        
        # Kelly formula
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply fraction and cap
        adjusted_kelly = kelly * self.fraction
        final_kelly = np.clip(adjusted_kelly, 0, self.max_fraction)
        
        stats = {
            "raw_kelly": kelly,
            "adjusted_kelly": adjusted_kelly,
            "final_kelly": final_kelly,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "n_trades": len(trades),
        }
        
        logger.debug(f"Kelly: {final_kelly:.2%} (raw: {kelly:.2%})")
        
        return final_kelly, stats

    def get_position_size(
        self,
        capital: float,
        current_price: float,
        confidence: float = 1.0,
    ) -> float:
        """Get position size in units.
        
        Args:
            capital: Available capital
            current_price: Current asset price
            confidence: Confidence multiplier (0-1)
            
        Returns:
            Position size in asset units
        """
        kelly, _ = self.calculate_kelly()
        adjusted_kelly = kelly * confidence
        
        dollar_amount = capital * adjusted_kelly
        position_units = dollar_amount / current_price
        
        return position_units


class RiskParity:
    """Risk Parity position sizing for multi-asset portfolios.
    
    Allocates positions so each asset contributes equally to portfolio risk.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        target_volatility: float = 0.02,  # 2% daily vol target
        min_weight: float = 0.05,
        max_weight: float = 0.30,
    ):
        """Initialize Risk Parity calculator.
        
        Args:
            lookback: Window for volatility calculation
            target_volatility: Target portfolio volatility (daily)
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
        """
        self.lookback = lookback
        self.target_volatility = target_volatility
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Price history per asset
        self.returns_history: dict[str, list[float]] = {}

    def update_prices(self, symbol: str, price: float) -> None:
        """Update price for an asset.
        
        Args:
            symbol: Asset symbol
            price: Current price
        """
        if symbol not in self.returns_history:
            self.returns_history[symbol] = []
            self._last_prices = {}
        
        if symbol in self._last_prices:
            ret = (price - self._last_prices[symbol]) / self._last_prices[symbol]
            self.returns_history[symbol].append(ret)
            self.returns_history[symbol] = self.returns_history[symbol][-self.lookback:]
        
        self._last_prices[symbol] = price

    def update_returns(self, symbol: str, returns: list[float]) -> None:
        """Directly update returns for an asset.
        
        Args:
            symbol: Asset symbol
            returns: List of historical returns
        """
        self.returns_history[symbol] = returns[-self.lookback:]

    def calculate_weights(self) -> dict[str, float]:
        """Calculate risk parity weights.
        
        Returns:
            Dict of symbol to weight
        """
        symbols = list(self.returns_history.keys())
        
        if len(symbols) == 0:
            return {}
        
        # Calculate volatility for each asset
        volatilities = {}
        for symbol in symbols:
            returns = self.returns_history[symbol]
            if len(returns) >= 10:
                volatilities[symbol] = np.std(returns) * np.sqrt(1440)  # Daily vol
            else:
                volatilities[symbol] = 1.0
        
        # Inverse volatility weighting
        total_inv_vol = sum(1.0 / (v + 1e-8) for v in volatilities.values())
        
        weights = {}
        for symbol, vol in volatilities.items():
            inv_vol_weight = (1.0 / (vol + 1e-8)) / total_inv_vol
            
            # Apply constraints
            constrained_weight = np.clip(inv_vol_weight, self.min_weight, self.max_weight)
            weights[symbol] = constrained_weight
        
        # Normalize to sum to 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        # Scale to target volatility
        portfolio_vol = self._estimate_portfolio_vol(weights)
        vol_scalar = self.target_volatility / (portfolio_vol + 1e-8)
        vol_scalar = np.clip(vol_scalar, 0.1, 2.0)
        
        scaled_weights = {k: v * vol_scalar for k, v in weights.items()}
        
        logger.debug(f"Risk parity weights: {scaled_weights}")
        
        return scaled_weights

    def _estimate_portfolio_vol(self, weights: dict[str, float]) -> float:
        """Estimate portfolio volatility.
        
        Args:
            weights: Asset weights
            
        Returns:
            Estimated portfolio volatility
        """
        symbols = list(weights.keys())
        n = len(symbols)
        
        if n == 0:
            return 0.0
        
        # Simple estimate (assumes no correlation)
        vol_sq = 0
        for symbol, weight in weights.items():
            returns = self.returns_history[symbol]
            if returns:
                asset_vol = np.std(returns) * np.sqrt(1440)
                vol_sq += (weight * asset_vol) ** 2
        
        return np.sqrt(vol_sq)

    def get_position_sizes(
        self,
        capital: float,
        current_prices: dict[str, float],
    ) -> dict[str, float]:
        """Get position sizes for all assets.
        
        Args:
            capital: Total available capital
            current_prices: Dict of symbol to current price
            
        Returns:
            Dict of symbol to position size (in units)
        """
        weights = self.calculate_weights()
        
        positions = {}
        for symbol, weight in weights.items():
            if symbol in current_prices:
                dollar_amount = capital * weight
                positions[symbol] = dollar_amount / current_prices[symbol]
            else:
                positions[symbol] = 0.0
        
        return positions


class PositionSizer:
    """Combined position sizing using multiple methods."""
    
    def __init__(
        self,
        method: str = "hybrid",
        kelly_fraction: float = 0.5,
        risk_parity_target_vol: float = 0.02,
        max_total_exposure: float = 1.0,
    ):
        """Initialize combined position sizer.
        
        Args:
            method: Sizing method ('kelly', 'risk_parity', 'hybrid')
            kelly_fraction: Kelly fraction for Kelly method
            risk_parity_target_vol: Target vol for risk parity
            max_total_exposure: Maximum total portfolio exposure
        """
        self.method = method
        self.max_total_exposure = max_total_exposure
        
        self.kelly = KellyCriterion(fraction=kelly_fraction)
        self.risk_parity = RiskParity(target_volatility=risk_parity_target_vol)

    def record_trade(self, pnl_pct: float) -> None:
        """Record trade for Kelly calculation."""
        self.kelly.record_trade(pnl_pct)

    def update_prices(self, symbol: str, price: float) -> None:
        """Update price for risk parity."""
        self.risk_parity.update_prices(symbol, price)

    def get_position_sizes(
        self,
        capital: float,
        current_prices: dict[str, float],
        agent_signals: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Get optimal position sizes.
        
        Args:
            capital: Available capital
            current_prices: Current prices per symbol
            agent_signals: Optional signals from RL agent (-1 to 1)
            
        Returns:
            Position sizes per symbol
        """
        if self.method == "kelly":
            kelly_fraction, _ = self.kelly.calculate_kelly()
            positions = {}
            for symbol, price in current_prices.items():
                signal = agent_signals.get(symbol, 0) if agent_signals else 0
                size = capital * kelly_fraction * abs(signal) / price
                positions[symbol] = size * np.sign(signal)
            return positions
        
        elif self.method == "risk_parity":
            return self.risk_parity.get_position_sizes(capital, current_prices)
        
        else:  # hybrid
            kelly_fraction, _ = self.kelly.calculate_kelly()
            rp_weights = self.risk_parity.calculate_weights()
            
            positions = {}
            for symbol, price in current_prices.items():
                rp_weight = rp_weights.get(symbol, 1.0 / len(current_prices))
                signal = agent_signals.get(symbol, 0) if agent_signals else 0
                
                # Combine Kelly and Risk Parity
                dollar_amount = capital * rp_weight * kelly_fraction * (0.5 + 0.5 * abs(signal))
                positions[symbol] = dollar_amount / price * np.sign(signal)
            
            # Apply exposure limit
            total_exposure = sum(abs(p * current_prices[s]) for s, p in positions.items()) / capital
            if total_exposure > self.max_total_exposure:
                scale = self.max_total_exposure / total_exposure
                positions = {k: v * scale for k, v in positions.items()}
            
            return positions
