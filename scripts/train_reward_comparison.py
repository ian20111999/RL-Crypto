"""快速訓練對比3種優化reward functions"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data import DataCollector, DataProcessor
from src.envs import MultiAssetTradingEnv


def create_env_with_reward(config, features, prices, reward_type):
    """創建使用指定reward function的環境"""
    env = MultiAssetTradingEnv(
        features_dict=features,
        prices_dict=prices,
        symbols=config["symbols"],
        initial_capital=config["trading"]["initial_capital"],
        leverage=config["trading"]["leverage"],
        maker_fee=config["trading"]["fees"]["maker"],
        taker_fee=config["trading"]["fees"]["taker"],
        slippage=config["trading"]["slippage"],
        transaction_penalty=config["environment"]["transaction_penalty"],
        episode_length=config["environment"]["episode_length"],
        reward_function_type=reward_type,  # 關鍵：指定reward
    )
    return env


def main():
    logger.info("="*80)
    logger.info("PPO Reward Function Optimization - Quick Test")
    logger.info("="*80)
    
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    #準備數據
    collector = DataCollector("config/config.yaml")
    processor = DataProcessor(lookback=config["data"]["lookback"])
    
    # 載入已訓練的processor參數
    params_path = Path(config["data"]["data_dir"]) / "processor_params.pkl"
    if params_path.exists():
        processor.load_params(str(params_path))
        logger.info("Loaded processor parameters")
    
    # Load and process data
    symbols = ["BTCUSDT"]  # 只用BTC避免ETH數據問題
    features_dict = {}
    prices_dict = {}
    
    for symbol in symbols:
        df = collector.load_symbol(symbol)
        split_idx = int(len(df) * 0.8)  # 固定80%訓練
        train_df = df.iloc[:split_idx].copy()
        
        # Process with loaded parameters
        processed_df = processor.process(train_df, fit=False)
        
        # Create sequences (正確方法)
        features, prices, _ = processor.create_sequences(processed_df)
        
        features_dict[symbol] = features
        prices_dict[symbol] = prices
    
    logger.info(f"Loaded {len(symbols)} symbols")
    logger.info(f"Training samples: {len(features_dict[symbols[0]])}")
    
    # 測試3種reward configurations
    reward_types = [
        "v1_aggressive",  # 激進型
        "v2_balanced",    # 平衡型  
        "v3_sharpe",      # Sharpe型
    ]
    
    results = {}
    
    for reward_type in reward_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training with reward: {reward_type}")
        logger.info(f"{'='*80}")
        
        # 創建環境
        env = create_env_with_reward(config, features_dict, prices_dict, reward_type)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        
        # 創建PPO模型（快速測試：較少steps）
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log=f"logs/ppo_{reward_type}",
        )
        
        # 快速訓練（100k steps測試）
        logger.info(f"Training for 100k steps...")
        model.learn(
            total_timesteps=100_000,
            progress_bar=True,
        )
        
        # 保存模型
        save_path = Path(f"models/ppo_{reward_type}_quick")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path / "model"))
        vec_env.save(str(save_path / "vecnormalize.pkl"))
        
        logger.info(f"✅ Saved to {save_path}")
        
        results[reward_type] = {
            "model_path": str(save_path),
            "steps": 100_000,
        }
    
    logger.info(f"\n{'='*80}")
    logger.info("Training completed for all reward types!")
    logger.info(f"{'='*80}")
    
    # 打印回測命令
    logger.info("\nNext step: Run backtest comparison")
    logger.info("```bash")
    for reward_type in reward_types:
        logger.info(f"python scripts/backtest.py --model models/ppo_{reward_type}_quick")
    logger.info("```")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
