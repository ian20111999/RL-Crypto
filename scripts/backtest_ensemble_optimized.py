"""優化後的ensemble回測腳本"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from stable_baselines3 import PPO, SAC, TD3

from src.data import DataCollector, DataProcessor
from src.backtesting import BacktestEngine
from src.backtesting.report import BacktestReport
from src.backtesting.backtest_engine import (
    buy_and_hold_strategy,
    sma_crossover_strategy,
    rsi_strategy,
)
from src.agents.dynamic_weights import DynamicWeightManager


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_optimized_ensemble(
    ppo_model, sac_model, td3_model,
    features, lookback,
    threshold=0.1,
    initial_weights=None,
):
    """
    創建優化ensemble：交易閾值 + 動態權重
    
    Args:
        threshold: 交易閾值（0.1 = 10%變化才交易）
        initial_weights: 初始權重
    """
    if initial_weights is None:
        initial_weights = {'ppo': 0.5, 'sac': 0.3, 'td3': 0.2}
    
    state = {"position": 0.0}
    weight_manager = DynamicWeightManager(
        ['ppo', 'sac', 'td3'],
        window_size=100,
        method='performance',
        initial_weights=initial_weights
    )
    
    def strategy(prices: pd.DataFrame, idx: int) -> float:
        if idx < lookback:
            return 0.0
        
        # Build observation
        n_assets = 2
        market_obs = np.zeros((n_assets, lookback, features.shape[1]), dtype=np.float32)
        
        for i in range(n_assets):
            start_idx = max(0, idx - lookback)
            window = features[start_idx:idx]
            
            if len(window) < lookback:
                padding = np.zeros((lookback - len(window), features.shape[1]), dtype=np.float32)
                window = np.vstack([padding, window])
            
            market_obs[i] = window
        
        positions = np.zeros(n_assets, dtype=np.float32)
        positions[0] = state["position"]
        portfolio_obs = np.concatenate([positions, [1.0 - abs(state["position"]), 1.0]]).astype(np.float32)
        
        obs = {"market": market_obs, "portfolio": portfolio_obs}
        
        # Get predictions
        action_ppo, _ = ppo_model.predict(obs, deterministic=True)
        action_sac, _ = sac_model.predict(obs, deterministic=True)
        action_td3, _ = td3_model.predict(obs, deterministic=True)
        
        pos_ppo = float(action_ppo[0]) if len(action_ppo.shape) > 0 else float(action_ppo)
        pos_sac = float(action_sac[0]) if len(action_sac.shape) > 0 else float(action_sac)
        pos_td3 = float(action_td3[0]) if len(action_td3.shape) > 0 else float(action_td3)
        
        # 動態權重
        weights = weight_manager.get_weights()
        new_position = (
            weights['ppo'] * pos_ppo +
            weights['sac'] * pos_sac +
            weights['td3'] * pos_td3
        )
        new_position = np.clip(new_position, -1.0, 1.0)
        
        # 核心優化：交易閾值
        position_change = abs(new_position - state["position"])
        if position_change < threshold:
            return state["position"]  # 不交易
        
        # 更新動態權重
        if idx > lookback:
            curr_price = prices["close"].iloc[idx]
            prev_price = prices["close"].iloc[idx - 1]
            price_change = (curr_price - prev_price) / prev_price
            
            rewards = {
                'ppo': pos_ppo * price_change,
                'sac': pos_sac * price_change,
                'td3': pos_td3 * price_change,
            }
            weight_manager.update(rewards)
        
        state["position"] = new_position
        return new_position
    
    return strategy


def main():
    logger.info("=" * 60)
    logger.info("Optimized Ensemble Backtest")
    logger.info("=" * 60)
    
    config = load_config()
    symbol = "BTCUSDT"
    
    # Load data
    collector = DataCollector("config/config.yaml")
    df = collector.load_symbol(symbol)
    
    prices = df[["open", "high", "low", "close", "volume"]].copy()
    prices.index = pd.to_datetime(df["open_time"])
    
    split_idx = int(len(prices) * 0.8  )
    test_prices = prices.iloc[split_idx:]
    
    logger.info(f"Backtest period: {test_prices.index[0]} to {test_prices.index[-1]}")
    logger.info(f"Number of bars: {len(test_prices)}")
    
    # Load ensemble models
    ensemble_dir = Path("models/ensemble")
    ppo_model = PPO.load(str(ensemble_dir / "ppo"))
    sac_model = SAC.load(str(ensemble_dir / "sac"))
    td3_model = TD3.load(str(ensemble_dir / "td3"))
    logger.info("Loaded ensemble models")
    
    # Pre-process features
    processor = DataProcessor(lookback=config["data"]["lookback"])
    params_path = Path(config["data"]["data_dir"]) / "processor_params.pkl"
    if params_path.exists():
        processor.load_params(str(params_path))
    
    processed_df = processor.process(test_prices.copy(), fit=False)
    features = processed_df[processor.feature_columns].values.astype(np.float32)
    logger.info(f"Pre-processed {len(features)} samples")
    
    # Create backtest engine
    engine = BacktestEngine(
        prices=test_prices,
        initial_capital=config["trading"]["initial_capital"],
        leverage=config["trading"]["leverage"],
        maker_fee=config["trading"]["fees"]["maker"],
        taker_fee=config["trading"]["fees"]["taker"],
        slippage=config["trading"]["slippage"],
    )
    
    # Define strategies
    strategies = {
        "Buy & Hold": buy_and_hold_strategy,
        "Ensemble-Original": create_optimized_ensemble(
            ppo_model, sac_model, td3_model, features, config["data"]["lookback"],
            threshold=0.0,  # 無閾值（原版）
        ),
        "Ensemble-t0.05": create_optimized_ensemble(
            ppo_model, sac_model, td3_model, features, config["data"]["lookback"],
            threshold=0.05,  # 5%閾值
        ),
        "Ensemble-t0.10": create_optimized_ensemble(
            ppo_model, sac_model, td3_model, features, config["data"]["lookback"],
            threshold=0.1,  # 10%閾值（推薦）
        ),
        "Ensemble-t0.20": create_optimized_ensemble(
            ppo_model, sac_model, td3_model, features, config["data"]["lookback"],
            threshold=0.2,  # 20%閾值
        ),
    }
    
    # Compare strategies
    logger.info("\nRunning backtest comparison...")
    comparison = engine.compare_strategies(strategies)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    print(comparison)
    logger.info("=" * 80)
    
    # Collect detailed results
    all_results = {}
    for name, strategy in strategies.items():
        engine.run(strategy, verbose=False)
        metrics = engine.get_metrics()
        if engine.results is not None:
            metrics['equity_curve'] = engine.results['portfolio_value'].tolist()
        all_results[name] = metrics
    
    # Generate report
    logger.info("\nGenerating detailed report...")
    report_gen = BacktestReport()
    report_gen.generate_full_report(
        all_results,
        timestamps=test_prices.index,
        save_html=True
    )
    
    logger.info("Optimization test completed!")


if __name__ == "__main__":
    main()
