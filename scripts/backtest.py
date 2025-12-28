"""Backtest script for RL-Crypto."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data import DataCollector, DataProcessor
from src.envs import MultiAssetTradingEnv
from src.backtesting import BacktestEngine
from src.backtesting.backtest_engine import (
    buy_and_hold_strategy,
    sma_crossover_strategy,
    rsi_strategy,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_rl_strategy(model, pre_processed_features: np.ndarray, lookback: int = 60, n_assets: int = 2):
    """Create a strategy function from trained RL model.

    Args:
        model: Trained PPO model
        pre_processed_features: Pre-processed feature array (n_samples, n_features)
        lookback: Lookback window
        n_assets: Number of assets in training

    Returns:
        Strategy function
    """
    # Track state for multi-step predictions
    state = {"position": 0.0, "cash_ratio": 1.0}
    
    def rl_strategy(prices: pd.DataFrame, idx: int) -> float:
        if idx < lookback:
            return 0.0
        
        # Use pre-processed features directly
        if idx >= len(pre_processed_features):
            return state["position"]

        try:
            # Get features for this window
            start_idx = max(0, idx - lookback + 1)
            features = pre_processed_features[start_idx:idx + 1]
            
            if len(features) < lookback:
                # Pad with zeros if needed
                padding = np.zeros((lookback - len(features), features.shape[1]), dtype=np.float32)
                features = np.vstack([padding, features])

            # Create market observation: (n_assets, lookback, n_features)
            market_obs = np.zeros((n_assets, lookback, features.shape[1]), dtype=np.float32)
            for i in range(n_assets):
                market_obs[i] = features
            
            # Portfolio state: [positions (n_assets), cash_ratio, portfolio_value_ratio]
            positions = np.zeros(n_assets, dtype=np.float32)
            positions[0] = state["position"]
            portfolio_obs = np.concatenate([
                positions,
                [state["cash_ratio"], 1.0]
            ]).astype(np.float32)

            obs = {
                "market": market_obs,
                "portfolio": portfolio_obs,
            }

            action, _ = model.predict(obs, deterministic=True)
            
            # Update state based on action
            new_position = float(action[0])
            state["position"] = np.clip(new_position, -1, 1)
            state["cash_ratio"] = max(0, 1 - abs(state["position"]))
            
            return new_position

        except Exception as e:
            logger.debug(f"RL strategy error at idx {idx}: {e}")
            return state["position"]

    return rl_strategy


def run_single_asset_backtest(
    config: dict,
    symbol: str,
    model_path: str = None,
    save_plot: bool = True,
) -> dict:
    """Run backtest for a single asset.

    Args:
        config: Configuration dict
        symbol: Trading symbol
        model_path: Path to trained model
        save_plot: Whether to save plots

    Returns:
        Dict of metrics
    """
    logger.info(f"Running backtest for {symbol}...")

    # Load data
    collector = DataCollector("config/config.yaml")
    df = collector.load_symbol(symbol)

    if df.empty:
        logger.error(f"No data for {symbol}")
        return {}

    # Prepare price DataFrame
    prices = df[["open", "high", "low", "close", "volume"]].copy()
    prices.index = pd.to_datetime(df["open_time"])

    # Split for out-of-sample backtest
    split_idx = int(len(prices) * 0.8)
    test_prices = prices.iloc[split_idx:]

    logger.info(f"Backtest period: {test_prices.index[0]} to {test_prices.index[-1]}")
    logger.info(f"Number of bars: {len(test_prices)}")

    # Create backtest engine
    engine = BacktestEngine(
        prices=test_prices,
        initial_capital=config["trading"]["initial_capital"],
        leverage=config["trading"]["leverage"],
        maker_fee=config["trading"]["fees"]["maker"],
        taker_fee=config["trading"]["fees"]["taker"],
        slippage=config["trading"]["slippage"],
    )

    # Define strategies to compare
    strategies = {
        "Buy & Hold": buy_and_hold_strategy,
        "SMA Crossover (10/50)": lambda p, i: sma_crossover_strategy(p, i, 10, 50),
        "SMA Crossover (20/100)": lambda p, i: sma_crossover_strategy(p, i, 20, 100),
        "RSI Mean Reversion": rsi_strategy,
    }

    # Add RL strategy if model available
    model_zip = Path(f"{model_path}.zip")
    if model_path and model_zip.exists():
        logger.info(f"Loading model from {model_path}")
        try:
            model = PPO.load(model_path)
            processor = DataProcessor(lookback=config["data"]["lookback"])

            # Load processor params
            params_path = Path(config["data"]["data_dir"]) / "processor_params.pkl"
            if params_path.exists():
                processor.load_params(str(params_path))
                logger.info(f"Loaded processor params from {params_path}")

            # Pre-process the entire test data
            logger.info("Pre-processing features for RL strategy...")
            processed_df = processor.process(test_prices.copy(), fit=False)
            if len(processed_df) > 0:
                pre_processed_features = processed_df[processor.feature_columns].values.astype(np.float32)
                logger.info(f"Pre-processed {len(pre_processed_features)} samples")
                
                strategies["PPO RL Agent"] = create_rl_strategy(
                    model, pre_processed_features, config["data"]["lookback"]
                )
                logger.info("Added PPO RL Agent strategy")
            else:
                logger.error("Failed to pre-process features")
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning(f"Model not found at {model_path}")

    # Compare strategies
    comparison = engine.compare_strategies(strategies)

    # Run detailed backtest for best strategy
    best_strategy_name = comparison["sharpe_ratio"].idxmax()
    logger.info(f"Best strategy by Sharpe: {best_strategy_name}")

    engine.run(strategies[best_strategy_name])
    engine.print_metrics()

    if save_plot:
        plot_path = f"logs/backtest_{symbol}.png"
        engine.plot_results(save_path=plot_path, show=False)

    return {
        "symbol": symbol,
        "comparison": comparison.to_dict(),
        "best_strategy": best_strategy_name,
        "best_metrics": engine.get_metrics(),
    }


def main():
    """Main backtest function."""
    parser = argparse.ArgumentParser(description="Backtest RL-Crypto strategies")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default="models/ppo_trading_final", help="Model path")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to backtest")
    parser.add_argument("--all", action="store_true", help="Backtest all configured symbols")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    # Setup logging
    logger.add(
        "logs/backtest_{time}.log",
        rotation="100 MB",
        level="INFO",
    )

    logger.info("=" * 60)
    logger.info("RL-Crypto Backtesting")
    logger.info("=" * 60)

    config = load_config(args.config)

    if args.symbol:
        # Backtest single symbol
        results = run_single_asset_backtest(
            config,
            args.symbol,
            model_path=args.model,
            save_plot=not args.no_plot,
        )
    elif args.all:
        # Backtest all symbols
        all_results = []
        for symbol in config["symbols"]:
            try:
                results = run_single_asset_backtest(
                    config,
                    symbol,
                    model_path=args.model,
                    save_plot=not args.no_plot,
                )
                all_results.append(results)
            except Exception as e:
                logger.error(f"Failed to backtest {symbol}: {e}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 60)
        for result in all_results:
            if result:
                metrics = result["best_metrics"]
                logger.info(
                    f"{result['symbol']}: "
                    f"Best={result['best_strategy']}, "
                    f"Return={metrics['total_return_pct']:.2f}%, "
                    f"Sharpe={metrics['sharpe_ratio']:.2f}"
                )
    else:
        # Default: backtest first symbol
        results = run_single_asset_backtest(
            config,
            config["symbols"][0],
            model_path=args.model,
            save_plot=not args.no_plot,
        )

    logger.info("Backtesting completed!")


if __name__ == "__main__":
    main()
