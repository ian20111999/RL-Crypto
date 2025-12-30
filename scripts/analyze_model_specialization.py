"""分析各模型在不同市場狀態下的專長"""

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


def detect_market_regimes(prices: pd.DataFrame, window=50):
    """
    檢測市場狀態並分類數據
    
    Returns:
        dict: {
            'high_volatility': indices,
            'trending_up': indices,
            'trending_down': indices,
            'ranging': indices,
        }
    """
    close = prices['close']
    
    # 計算波動率（滾動標準差）
    volatility = close.pct_change().rolling(window).std()
    
    # 計算趨勢（SMA斜率）
    sma = close.rolling(window).mean()
    trend = (sma - sma.shift(window)) / sma.shift(window)
    
    # 分類閾值
    HIGH_VOL_THRESHOLD = volatility.quantile(0.75)  # 前25%高波動
    TREND_THRESHOLD = 0.02  # 2%趨勢
    
    # 分類
    regimes = {
        'high_volatility': [],
        'trending_up': [],
        'trending_down': [],
        'ranging': [],
    }
    
    for i in range(len(prices)):
        if i < window:
            continue
        
        vol = volatility.iloc[i]
        tr = trend.iloc[i]
        
        # 優先級：高波動 > 趨勢 > 橫盤
        if vol > HIGH_VOL_THRESHOLD:
            regimes['high_volatility'].append(i)
        elif tr > TREND_THRESHOLD:
            regimes['trending_up'].append(i)
        elif tr < -TREND_THRESHOLD:
            regimes['trending_down'].append(i)
        else:
            regimes['ranging'].append(i)
    
    # 統計
    total = sum(len(v) for v in regimes.values())
    logger.info("Market regime distribution:")
    for regime, indices in regimes.items():
        pct = len(indices) / total * 100 if total > 0 else 0
        logger.info(f"  {regime}: {len(indices)} bars ({pct:.1f}%)")
    
    return regimes


def backtest_model_on_regime(
    model,
    prices: pd.DataFrame,
    features: np.ndarray,
    regime_indices: list,
    config: dict,
):
    """在特定市場狀態下回測模型"""
    if len(regime_indices) < 100:
        logger.warning(f"Too few samples ({len(regime_indices)}), skipping")
        return None
    
    # 過濾超出範圍的索引
    valid_indices = [i for i in regime_indices if i < len(prices) and i < len(features)]
    if len(valid_indices) < 100:
        logger.warning(f"Too few valid samples ({len(valid_indices)}), skipping")
        return None
    
    # 創建子集數據
    regime_prices = prices.iloc[valid_indices].copy()
    regime_features = features[valid_indices]
    
    # 創建回測引擎
    engine = BacktestEngine(
        prices=regime_prices,
        initial_capital=config["trading"]["initial_capital"],
        leverage=config["trading"]["leverage"],
        maker_fee=config["trading"]["fees"]["maker"],
        taker_fee=config["trading"]["fees"]["taker"],
        slippage=config["trading"]["slippage"],
    )
    
    # 創建策略
    lookback = config["data"]["lookback"]
    
    def rl_strategy(prices_df, idx):
        if idx < lookback:
            return 0.0
        
        # Build observation
        market_obs = np.zeros((2, lookback, regime_features.shape[1]), dtype=np.float32)
        start_idx = max(0, idx - lookback)
        window = regime_features[start_idx:idx]
        
        if len(window) < lookback:
            padding = np.zeros((lookback - len(window), regime_features.shape[1]), dtype=np.float32)
            window = np.vstack([padding, window])
        
        market_obs[0] = window
        market_obs[1] = window
        
        # portfolio_obs: [pos1, pos2, cash_ratio, leverage_used]
        portfolio_obs = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        obs = {"market": market_obs, "portfolio": portfolio_obs}
        
        action, _ = model.predict(obs, deterministic=True)
        position = float(action[0]) if len(action.shape) > 0 else float(action)
        return np.clip(position, -1.0, 1.0)
    
    # 運行回測
    engine.run(rl_strategy, verbose=False)
    metrics = engine.get_metrics()
    
    return metrics


def main():
    logger.info("=" * 80)
    logger.info("Model Specialization Analysis")
    logger.info("=" * 80)
    
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Load data
    collector = DataCollector("config/config.yaml")
    df = collector.load_symbol("BTCUSDT")
    
    prices = df[["open", "high", "low", "close", "volume"]].copy()
    prices.index = pd.to_datetime(df["open_time"])
    
    # Use test split
    split_idx = int(len(prices) * 0.8)
    test_prices = prices.iloc[split_idx:].copy()
    
    logger.info(f"Test period: {test_prices.index[0]} to {test_prices.index[-1]}")
    logger.info(f"Total bars: {len(test_prices)}")
    
    # Pre-process features
    processor = DataProcessor(lookback=config["data"]["lookback"])
    params_path = Path(config["data"]["data_dir"]) / "processor_params.pkl"
    if params_path.exists():
        processor.load_params(str(params_path))
    
    processed_df = processor.process(test_prices.copy(), fit=False)
    features = processed_df[processor.feature_columns].values.astype(np.float32)
    logger.info(f"Pre-processed {len(features)} samples")
    
    # Detect market regimes
    logger.info("\nDetecting market regimes...")
    regimes = detect_market_regimes(test_prices)
    
    # Load models
    logger.info("\nLoading models...")
    ensemble_dir = Path("models/ensemble")
    models = {
        'PPO': PPO.load(str(ensemble_dir / "ppo")),
        'SAC': SAC.load(str(ensemble_dir / "sac")),
        'TD3': TD3.load(str(ensemble_dir / "td3")),
    }
    
    # Test each model on each regime
    logger.info("\n" + "=" * 80)
    logger.info("Testing models on different market regimes...")
    logger.info("=" * 80)
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"\nTesting {model_name}...")
        results[model_name] = {}
        
        for regime_name, indices in regimes.items():
            if len(indices) < 100:
                continue
            
            logger.info(f"  {regime_name}: {len(indices)} bars...")
            metrics = backtest_model_on_regime(
                model, test_prices, features, indices, config
            )
            
            if metrics:
                results[model_name][regime_name] = metrics
                logger.info(f"    Return: {metrics['total_return_pct']:.2f}%")
                logger.info(f"    Sharpe: {metrics['sharpe_ratio']:.2f}")
                logger.info(f"    Trades: {metrics['n_trades']}")
    
    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("SPECIALIZATION ANALYSIS")
    logger.info("=" * 80)
    
    # Find best model for each regime
    for regime_name in regimes.keys():
        if len(regimes[regime_name]) < 100:
            continue
        
        logger.info(f"\n{regime_name.upper()}:")
        regime_results = {
            model: results[model].get(regime_name, {})
            for model in models.keys()
        }
        
        # Sort by return
        sorted_models = sorted(
            regime_results.items(),
            key=lambda x: x[1].get('total_return_pct', -999),
            reverse=True
        )
        
        for rank, (model, metrics) in enumerate(sorted_models, 1):
            if metrics:
                ret = metrics['total_return_pct']
                sharpe = metrics['sharpe_ratio']
                trades = metrics['n_trades']
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                logger.info(f"  {emoji} {model}: {ret:+.2f}% | Sharpe: {sharpe:.2f} | Trades: {trades}")
    
    # Recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDED WEIGHTS")
    logger.info("=" * 80)
    
    recommendations = {}
    for regime_name in regimes.keys():
        if len(regimes[regime_name]) < 100:
            continue
        
        # Find best performing model
        best_model = None
        best_return = -999
        
        for model_name in models.keys():
            if regime_name in results[model_name]:
                ret = results[model_name][regime_name]['total_return_pct']
                if ret > best_return:
                    best_return = ret
                    best_model = model_name
        
        if best_model:
            # Assign high weight to best model
            weights = {m: 0.1 for m in models.keys()}
            weights[best_model] = 0.8
            recommendations[regime_name] = weights
            
            logger.info(f"\n{regime_name}:")
            for model, weight in weights.items():
                logger.info(f"  {model}: {weight:.1f}")
    
    # Save recommendations
    import json
    output_file = "models/ensemble/regime_weights.json"
    with open(output_file, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logger.info(f"\n✅ Saved recommendations to {output_file}")
    logger.info("\nAnalysis completed!")


if __name__ == "__main__":
    main()
