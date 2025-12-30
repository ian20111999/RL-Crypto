"""對比PPO-Stability vs 原版PPO"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from stable_baselines3 import PPO

from src.data import DataCollector, DataProcessor
from src.backtesting import BacktestEngine


def main():
    logger.info("="*80)
    logger.info("PPO-Stability vs Original PPO Comparison")
    logger.info("="*80)
    
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Data
    collector = DataCollector("config/config.yaml")
    processor = DataProcessor(lookback=config["data"]["lookback"])
    
    params_path = Path(config["data"]["data_dir"]) / "processor_params.pkl"
    if params_path.exists():
        processor.load_params(str(params_path))
    
    # Load BTCUSDT test data
    df = collector.load_symbol("BTCUSDT")
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    test_processed = processor.process(test_df, fit=False)
    features, prices, _ = processor.create_sequences(test_processed)
    
    logger.info(f"Test samples: {len(features)}")
    
    # Create engines
    engine1 = BacktestEngine(
        prices=test_df[["open", "high", "low", "close", "volume"]],
        initial_capital=config["trading"]["initial_capital"],
        leverage=config["trading"]["leverage"],
        maker_fee=config["trading"]["fees"]["maker"],
        taker_fee=config["trading"]["fees"]["taker"],
        slippage=config["trading"]["slippage"],
    )
    
    engine2 = BacktestEngine(
        prices=test_df[["open", "high", "low", "close", "volume"]],
        initial_capital=config["trading"]["initial_capital"],
        leverage=config["trading"]["leverage"],
        maker_fee=config["trading"]["fees"]["maker"],
        taker_fee=config["trading"]["fees"]["taker"],
        slippage=config["trading"]["slippage"],
    )
    
    # Load models
    logger.info("\nLoading models...")
    model_stability = PPO.load("models/ppo_stability/final_model")
    model_original = PPO.load("models/ppo_trading_final")
    
    logger.info("✅ Models loaded")
    
    # Create strategies
    lookback = config["data"]["lookback"]
    
    def create_strategy(model, features):
        def strategy(prices_df, idx):
            if idx < lookback:
                return 0.0
            
            # Simple observation
            market_obs = np.zeros((2, lookback, features.shape[1]), dtype=np.float32)
            
            start_idx = max(0, idx - lookback)
            window = features[start_idx:idx]
            
            if len(window) < lookback:
                padding = np.zeros((lookback - len(window), features.shape[1]), dtype=np.float32)
                window = np.vstack([padding, window])
            
            market_obs[0] = window
            market_obs[1] = window
            
            portfolio_obs = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
            obs = {"market": market_obs, "portfolio": portfolio_obs}
            
            action, _ = model.predict(obs, deterministic=True)
            position = float(action[0]) if len(action.shape) > 0 else float(action)
            return np.clip(position, -1.0, 1.0)
        
        return strategy
    
    stability_strategy = create_strategy(model_stability, features)
    original_strategy = create_strategy(model_original, features)
    
    # Run backtests
    logger.info("\nRunning PPO-Stability backtest...")
    engine1.run(stability_strategy, verbose=False)
    metrics_stability = engine1.get_metrics()
    
    logger.info("\nRunning PPO-Original backtest...")
    engine2.run(original_strategy, verbose=False)
    metrics_original = engine2.get_metrics()
    
    # Results
    logger.info("\n" + "="*80)
    logger.info("COMPARISON RESULTS")
    logger.info("="*80)
    
    comparison = pd.DataFrame({
        'PPO-Stability': {
            'Return (%)': metrics_stability['total_return_pct'],
            'Sharpe': metrics_stability['sharpe_ratio'],
            'Max DD (%)': metrics_stability['max_drawdown_pct'],
            'Trades': metrics_stability['n_trades'],
            'Win Rate (%)': metrics_stability.get('win_rate', 0) * 100,
        },
        'PPO-Original': {
            'Return (%)': metrics_original['total_return_pct'],
            'Sharpe': metrics_original['sharpe_ratio'],
            'Max DD (%)': metrics_original['max_drawdown_pct'],
            'Trades': metrics_original['n_trades'],
            'Win Rate (%)': metrics_original.get('win_rate', 0) * 100,
        }
    })
    
    comparison['Δ (Stab - Orig)'] = comparison['PPO-Stability'] - comparison['PPO-Original']
    
    print("\n", comparison.T)
    
    # Analysis
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS")
    logger.info("="*80)
    
    # Stability check: std of returns
    if 'portfolio_value' in engine1.results:
        pv1 = engine1.results['portfolio_value']
        pv2 = engine2.results['portfolio_value']
        
        returns1 = pd.Series(pv1).pct_change().dropna()
        returns2 = pd.Series(pv2).pct_change().dropna()
        
        logger.info(f"\nReturn Volatility (lower = more stable):")
        logger.info(f"  PPO-Stability: {returns1.std():.6f}")
        logger.info(f"  PPO-Original:  {returns2.std():.6f}")
        logger.info(f"  Improvement:   {((returns2.std() - returns1.std()) / returns2.std() * 100):.2f}%")
    
    # Verdict
    logger.info("\n" + "="*80)
    logger.info("VERDICT")
    logger.info("="*80)
    
    if metrics_stability['sharpe_ratio'] > metrics_original['sharpe_ratio']:
        logger.info("✅ PPO-Stability has BETTER risk-adjusted return (Sharpe)")
    else:
        logger.info("❌ PPO-Stability has WORSE risk-adjusted return")
    
    if metrics_stability['max_drawdown_pct'] < metrics_original['max_drawdown_pct']:
        logger.info("✅ PPO-Stability has LOWER max drawdown (better risk control)")
    else:
        logger.info("❌ PPO-Stability has HIGHER max drawdown")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
