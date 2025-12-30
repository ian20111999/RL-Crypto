"""訓練PPO-Stability模型"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from src.data import DataCollector, DataProcessor
from src.envs import MultiAssetTradingEnv
from src.envs.wrappers import create_stable_env


def main():
    logger.info("="*80)
    logger.info("PPO Stability Training")
    logger.info("="*80)
    
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Data
    collector = DataCollector("config/config.yaml")
    processor = DataProcessor(lookback=config["data"]["lookback"])
    
    # Load processor params
    params_path = Path(config["data"]["data_dir"]) / "processor_params.pkl"
    if params_path.exists():
        processor.load_params(str(params_path))
        logger.info("Loaded processor parameters")
    
    # Load data (只用BTCUSDT)
    logger.info("Loading BTCUSDT data...")
    df = collector.load_symbol("BTCUSDT")
    
    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    eval_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Train: {len(train_df)}, Eval: {len(eval_df)}")
    
    # Process
    train_processed = processor.process(train_df, fit=False)
    eval_processed = processor.process(eval_df, fit=False)
    
    # Create sequences
    train_features, train_prices, _ = processor.create_sequences(train_processed)
    eval_features, eval_prices, _ = processor.create_sequences(eval_processed)
    
    logger.info(f"Train samples: {len(train_features)}")
    logger.info(f"Eval samples: {len(eval_features)}")
    
    # Create environments with STABILITY reward
    trading_config = config["trading"]
    
    train_base_env = MultiAssetTradingEnv(
        features_dict={"BTCUSDT": train_features},
        prices_dict={"BTCUSDT": train_prices},
        symbols=["BTCUSDT"],
        initial_capital=trading_config["initial_capital"],
        leverage=trading_config["leverage"],
        maker_fee=trading_config["fees"]["maker"],
        taker_fee=trading_config["fees"]["taker"],
        slippage=trading_config["slippage"],
        transaction_penalty=config["environment"]["transaction_penalty"],
        episode_length=config["environment"]["episode_length"],
        reward_function_type="stability",  # 使用穩定性reward
    )
    
    eval_base_env = MultiAssetTradingEnv(
        features_dict={"BTCUSDT": eval_features},
        prices_dict={"BTCUSDT": eval_prices},
        symbols=["BTCUSDT"],
        initial_capital=trading_config["initial_capital"],
        leverage=trading_config["leverage"],
        maker_fee=trading_config["fees"]["maker"],
        taker_fee=trading_config["fees"]["taker"],
        slippage=trading_config["slippage"],
        transaction_penalty=config["environment"]["transaction_penalty"],
        episode_length=min(len(eval_features) - 1, config["environment"]["episode_length"]),
        reward_function_type="stability",
    )
    
    # 不使用wrapper，乾淨測試reward效果
    train_env = train_base_env
    eval_env = eval_base_env
    
    # # 原版：應用Action Smoothing Wrapper (已移除，避免兼容性問題)
    # train_env = create_stable_env(train_base_env, {
    #     'action_smoothing_alpha': 0.7,
    #     'enable_volatility_scaling': True,
    # })
    # 
    # eval_env = create_stable_env(eval_base_env, {
    #     'action_smoothing_alpha': 0.7,
    #     'enable_volatility_scaling': True,
    # })
    
    # Wrap
    train_vec_env = DummyVecEnv([lambda: train_env])
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
    
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, training=False)
    
    # Create PPO with CONSERVATIVE hyperparameters
    logger.info("\nCreating PPO model with stability hyperparameters...")
    model = PPO(
        "MultiInputPolicy",
        train_vec_env,
        learning_rate=3e-4,  # 會線性衰減
        n_steps=2048,
        batch_size=128,  # 較大batch更穩定
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # 保守clip避免劇烈更新
        ent_coef=0.01,   # 適度探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="logs/ppo_stability",
        device="auto",
    )
    
    # Callbacks
    save_path = Path("models/ppo_stability")
    save_path.mkdir(parents=True, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=str(save_path / "best"),
        log_path=str(save_path / "eval_logs"),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(save_path / "checkpoints"),
        name_prefix="ppo_stability",
    )
    
    # Train
    logger.info("\nTraining PPO-Stability for 1M steps...")
    model.learn(
        total_timesteps=1_000_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    # Save final model
    model.save(str(save_path / "final_model"))
    train_vec_env.save(str(save_path / "vecnormalize.pkl"))
    
    logger.info(f"\n✅ Training completed! Model saved to {save_path}")
    logger.info("\nNext: Run backtest comparison")
    logger.info(f"  python scripts/backtest.py --model {save_path}")


if __name__ == "__main__":
    main()
