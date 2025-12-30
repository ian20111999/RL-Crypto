"""Training script for multi-agent ensemble (PPO + SAC + TD3)."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from loguru import logger

from src.data import DataCollector, DataProcessor
from src.envs import MultiAssetTradingEnv
from src.agents.ensemble import MultiAgentEnsemble


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(config: dict, force_download: bool = False) -> tuple:
    """Prepare training and evaluation data."""
    logger.info("Preparing data...")

    collector = DataCollector("config/config.yaml")
    processor = DataProcessor(lookback=config["data"]["lookback"])

    # Load or collect data
    data_dir = Path(config["data"]["data_dir"])
    if not data_dir.exists() or force_download:
        logger.info("Collecting data from Binance...")
        raw_data = collector.collect_all()
    else:
        logger.info("Loading cached data...")
        raw_data = collector.load_all()

    # Process data
    logger.info("Processing features...")
    processed_data = processor.process_multiple(raw_data, fit=True)

    # Save processor params
    if processor._fitted:
        params_path = data_dir / "processor_params.pkl"
        processor.save_params(str(params_path))

    # Create sequences
    symbols = config["symbols"]
    train_features_dict = {}
    train_prices_dict = {}
    eval_features_dict = {}
    eval_prices_dict = {}

    train_ratio = 0.8

    for symbol in symbols:
        if symbol not in processed_data or processed_data[symbol].empty:
            logger.warning(f"No data for {symbol}, skipping")
            continue

        df = processed_data[symbol]
        features, prices, _ = processor.create_sequences(df)

        split_idx = int(len(features) * train_ratio)

        train_features_dict[symbol] = features[:split_idx]
        train_prices_dict[symbol] = prices[:split_idx]
        eval_features_dict[symbol] = features[split_idx:]
        eval_prices_dict[symbol] = prices[split_idx:]

        logger.info(f"{symbol}: {split_idx} train / {len(features) - split_idx} eval samples")

    return train_features_dict, train_prices_dict, eval_features_dict, eval_prices_dict


def create_environment(
    config: dict,
    features: dict,
    prices: dict,
) -> MultiAssetTradingEnv:
    """Create trading environment."""
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
    )
    return env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ensemble of RL agents")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--steps", type=int, default=100000, help="Training steps per agent")
    parser.add_argument("--force-download", action="store_true", help="Force re-download data")
    parser.add_argument("--voting", type=str, default="weighted_average", 
                       choices=["average", "weighted_average", "majority"],
                       help="Voting strategy")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RL-Crypto Ensemble Training")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)

    # Prepare data
    train_features, train_prices, eval_features, eval_prices = prepare_data(
        config, args.force_download
    )

    # Create environments
    logger.info("Creating training environment...")
    train_env = create_environment(config, train_features, train_prices)
    
    logger.info("Creating evaluation environment...")
    eval_env = create_environment(config, eval_features, eval_prices)

    # Create ensemble
    logger.info(f"Creating ensemble with {args.voting} voting...")
    ensemble = MultiAgentEnsemble(
        env=train_env,
        config_path=args.config,
        voting_strategy=args.voting,
    )

    # Add agents
    logger.info("Adding agents to ensemble...")
    
    # PPO
    ensemble.add_agent(
        name="ppo",
        algo="ppo",
        weight=1.0,
        learning_rate=config["ppo"]["learning_rate"],
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        ent_coef=config["ppo"]["ent_coef"],
    )
    
    # SAC
    ensemble.add_agent(
        name="sac",
        algo="sac",
        weight=1.0,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
    )
    
    # TD3
    ensemble.add_agent(
        name="td3",
        algo="td3",
        weight=1.0,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
    )

    logger.info("Starting ensemble training...")
    logger.info(f"Training {len(ensemble.agents)} agents for {args.steps} steps each")
    
    # Train all agents
    ensemble.train_all(total_timesteps=args.steps, progress_bar=True)

    # Save ensemble
    save_path = "models/ensemble"
    logger.info(f"Saving ensemble to {save_path}...")
    ensemble.save_all(save_path)

    # Evaluate
    logger.info("Evaluating ensemble...")
    logger.info("Running evaluation on eval environment...")
    
    # Reset environment - handle both Gym and SB3 API
    reset_result = eval_env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result  # Gym API
    else:
        obs = reset_result  # SB3 VecEnv API
    
    done = False
    total_reward = 0
    episode_length = 0
    
    while not done:
        action, info = ensemble.predict(obs, deterministic=True)
        
        # Step environment - handle both API styles
        step_result = eval_env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result  # Gym API
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result  # Old API
            
        total_reward += reward
        episode_length += 1
    
    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    logger.info(f"  Total Reward: {total_reward:.2f}")
    logger.info(f"  Episode Length: {episode_length}")
    logger.info(f"  Final Portfolio Value: {eval_env.portfolio_value:.2f}")
    logger.info(f"  Return: {(eval_env.portfolio_value / config['trading']['initial_capital'] - 1) * 100:.2f}%")
    logger.info("=" * 60)
    
    # Performance summary
    perf = ensemble.get_performance_summary()
    logger.info("\nEnsemble Performance Summary:")
    for agent_name, metrics in perf.items():
        logger.info(f"  {agent_name}: {metrics}")

    logger.info("\nTraining completed!")


if __name__ == "__main__":
    main()
