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
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data import DataCollector, DataProcessor
from src.envs import MultiAssetTradingEnv
from src.backtesting import BacktestEngine
from src.backtesting.report import BacktestReport
from src.agents.ensemble import MultiAgentEnsemble
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
            
            # Action Smoothing (Post-processing)
            alpha = 0.8 
            raw_action = float(action[0])
            smoothed_action = alpha * state.get("last_action", 0.0) + (1 - alpha) * raw_action
            state["last_action"] = smoothed_action
            
            # Deadband: preventing small trades
            threshold = 0.05  # 5% change required to trade
            current_position = state["position"]
            
            if abs(smoothed_action - current_position) < threshold:
                new_position = current_position
            else:
                new_position = np.clip(smoothed_action, -1, 1)

            # Update state based on action
            state["position"] = new_position
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
    use_full_data: bool = False,
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

    # Split for out-of-sample backtest or use full data
    if use_full_data:
        split_idx = 0
        logger.info("Using FULL dataset for backtesting")
    else:
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

    # Strategy List
    strategies = {
        "Buy & Hold": buy_and_hold_strategy,
        "SMA Crossover (10/50)": lambda p, i: sma_crossover_strategy(p, i, 10, 50),
        "SMA Crossover (20/100)": lambda p, i: sma_crossover_strategy(p, i, 20, 100),
        "RSI Mean Reversion": rsi_strategy,
    }

    # Add RL strategy if model available
    model_zip = Path(f"{model_path}.zip")
    ensemble_dir = Path(model_path)
    
    # Check for ensemble models
    if ensemble_dir.is_dir() and (ensemble_dir / "ppo.zip").exists():
        logger.info(f"Loading ensemble from {model_path}")
        try:
            from src.agents.ensemble import MultiAgentEnsemble
            
            # Create dummy env (required by ensemble but not used in backtest)
            processor = DataProcessor(lookback=config["data"]["lookback"])
            params_path = Path(config["data"]["data_dir"]) / "processor_params.pkl"
            if params_path.exists():
                processor.load_params(str(params_path))
            
            # Pre-process data for ensemble
            logger.info("Pre-processing features for Ensemble...")
            processed_df = processor.process(test_prices.copy(), fit=False)
            if len(processed_df) > 0:
                pre_processed_features = processed_df[processor.feature_columns].values.astype(np.float32)
                logger.info(f"Pre-processed {len(pre_processed_features)} samples")
                
                # Create ensemble strategy (similar to RL strategy but uses ensemble.predict)
                # We'll load models manually for prediction
                
                # Load individual models
                ppo_model = PPO.load(str(ensemble_dir / "ppo"))
                sac_model = SAC.load(str(ensemble_dir / "sac"))
                td3_model = TD3.load(str(ensemble_dir / "td3"))
                
                def create_ensemble_strategy(ppo, sac, td3, features, lookback, n_assets=2):
                    state = {"position": 0.0, "cash_ratio": 1.0}
                    prev_positions = np.zeros(n_assets)
                    
                    def ensemble_strategy(prices: pd.DataFrame, idx: int) -> float:
                        if idx < lookback:
                            return 0.0
                        
                        # Build observation - fix indexing to get exactly lookback samples
                        market_obs = np.zeros((n_assets, lookback, features.shape[1]), dtype=np.float32)
                        for i in range(n_assets):
                            # Get the lookback window ending at idx
                            start_idx = max(0, idx - lookback)
                            end_idx = idx
                            window = features[start_idx:end_idx]
                            
                            # Pad if necessary
                            if len(window) < lookback:
                                padding = np.zeros((lookback - len(window), features.shape[1]), dtype=np.float32)
                                window = np.vstack([padding, window])
                            
                            market_obs[i] = window
                        
                        positions = np.zeros(n_assets, dtype=np.float32)
                        positions[0] = state["position"]
                        portfolio_obs = np.concatenate([positions, [state["cash_ratio"], 1.0]]).astype(np.float32)
                        
                        obs = {"market": market_obs, "portfolio": portfolio_obs}
                        
                        # Get predictions from all three models
                        action_ppo, _ = ppo.predict(obs, deterministic=True)
                        action_sac, _ = sac.predict(obs, deterministic=True)
                        action_td3, _ = td3.predict(obs, deterministic=True)
                        
                        # Weighted average (equal weights)
                        action = (action_ppo + action_sac + action_td3) / 3.0
                        
                        # Extract position for first asset (BTCUSDT)
                        position = float(action[0]) if len(action.shape) > 0 else float(action)
                        position = np.clip(position, -1.0, 1.0)
                        
                        state["position"] = position
                        return position
                    
                    return ensemble_strategy
                
                strategies["Ensemble (PPO+SAC+TD3)"] = create_ensemble_strategy(
                    ppo_model, sac_model, td3_model, pre_processed_features, config["data"]["lookback"]
                )
                logger.info("Added Ensemble strategy")
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            import traceback
            traceback.print_exc()
    
    # Check for single PPO model
    elif model_path and model_zip.exists():
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
                
                n_assets = 1 if "stability" in str(model_path) else 2
                strategies["PPO RL Agent"] = create_rl_strategy(
                    model, pre_processed_features, config["data"]["lookback"], n_assets=n_assets
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

    # Collect all strategy results with equity curves
    all_strategy_results = {}
    for strategy_name in strategies.keys():
        # Run each strategy to get equity curve
        engine.run(strategies[strategy_name])
        metrics = engine.get_metrics()
        # Get equity curve from results
        if engine.results is not None:
            metrics['equity_curve'] = engine.results['portfolio_value'].tolist()
        else:
            metrics['equity_curve'] = []
        all_strategy_results[strategy_name] = metrics

    return {
        "symbol": symbol,
        "comparison": comparison.to_dict(),
        "best_strategy": best_strategy_name,
        "best_metrics": engine.get_metrics(),
        "all_strategies": all_strategy_results,
        "timestamps": test_prices.index,  # Add timestamps
    }


def main():
    """Main backtest function."""
    parser = argparse.ArgumentParser(description="Backtest RL-Crypto strategies")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default="models/ppo_trading_final", help="Model path")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to backtest")
    parser.add_argument("--all", action="store_true", help="Backtest all configured symbols")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--ensemble", type=str, default=None, help="Path to ensemble models")
    parser.add_argument("--report", action="store_true", help="Generate detailed HTML report")
    parser.add_argument("--full-data", action="store_true", help="Use full dataset for backtesting")
    parser.add_argument("--save-json", type=str, default=None, help="Save backtest results to JSON file")
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
            model_path=args.ensemble or args.model,
            save_plot=not args.no_plot,
            use_full_data=args.full_data,
        )
        
        # Generate detailed report if requested
        if args.report and results:
            logger.info("\nGenerating detailed report...")
            report_gen = BacktestReport()
            
            # Prepare results for report
            strategy_results = {}
            for strategy_name, metrics in results.items():
                if strategy_name not in ["symbol", "best_strategy", "best_metrics"]:
                    strategy_results[strategy_name] = metrics
            
            report_gen.generate_full_report(strategy_results, save_html=True)
            
    elif args.all:
        # Backtest all symbols
        all_results = []
        for symbol in config["symbols"]:
            try:
                results = run_single_asset_backtest(
                    config,
                    symbol,
                    model_path=args.ensemble or args.model,
                    save_plot=not args.no_plot,
                    use_full_data=args.full_data,
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
        # PPO Stability 模型只跑 BTCUSDT
        if args.model and "stability" in args.model:
            logger.info("⚠️ Detected Stability Model: Testing BTCUSDT only.")
            symbols_to_test = ["BTCUSDT"]
        else:
            symbols_to_test = [config["symbols"][0]]

        # Default: backtest first symbol (or BTCUSDT for stability)
        results = run_single_asset_backtest(
            config,
            symbols_to_test[0],
            model_path=args.ensemble or args.model,
            save_plot=not args.no_plot,
            use_full_data=args.full_data,
        )
        
        # Generate detailed report if requested
        if args.report and results:
            logger.info("\nGenerating detailed report...")
            report_gen = BacktestReport()
            
            # Use all_strategies data with equity curves
            strategy_results = results.get("all_strategies", {})
            timestamps = results.get("timestamps", None)
            
            if strategy_results:
                report_gen.generate_full_report(
                    strategy_results, 
                    timestamps=timestamps,
                    save_html=True
                )
            else:
                logger.warning("No strategy results available for report")
    
    # Save results to JSON if requested
    if args.save_json and results:
        import json
        
        json_output = {
            "symbol": results.get("symbol", "N/A"),
            "best_strategy": results.get("best_strategy", "N/A"),
            "sharpe_ratio": results.get("best_metrics", {}).get("sharpe_ratio", 0),
            "total_return_pct": results.get("best_metrics", {}).get("total_return_pct", 0),
            "max_drawdown_pct": results.get("best_metrics", {}).get("max_drawdown_pct", 0),
            "win_rate": results.get("best_metrics", {}).get("win_rate", 0),
            "volatility": results.get("best_metrics", {}).get("volatility", 0),
            "total_trades": results.get("best_metrics", {}).get("total_trades", 0),
        }
        
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        logger.info(f"Results saved to {json_path}")

    logger.info("Backtesting completed!")


if __name__ == "__main__":
    main()
