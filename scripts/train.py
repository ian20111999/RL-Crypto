"""Training script for RL-Crypto PPO agent."""

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
from src.agents import PPOAgent
from src.agents.callbacks import CheckpointCallback, EarlyStoppingCallback


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(config: dict, force_download: bool = False) -> tuple:
    """Prepare training and evaluation data.

    Args:
        config: Configuration dict
        force_download: Force re-download of data

    Returns:
        Tuple of (train_features, train_prices, eval_features, eval_prices)
    """
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

    # Save processor params if fitted
    if processor._fitted:
        params_path = data_dir / "processor_params.pkl"
        processor.save_params(str(params_path))

    # Create sequences for each symbol
    symbols = config["symbols"]
    train_features_dict = {}
    train_prices_dict = {}
    eval_features_dict = {}
    eval_prices_dict = {}

    # Train/eval split (80/20)
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


def create_environments(
    config: dict,
    train_features: dict,
    train_prices: dict,
    eval_features: dict,
    eval_prices: dict,
) -> tuple:
    """Create training and evaluation environments.

    Returns:
        Tuple of (train_env, eval_env)
    """
    symbols = list(train_features.keys())

    trading_config = config["trading"]
    env_config = config["environment"]

    # Training environment
    train_env = MultiAssetTradingEnv(
        features_dict=train_features,
        prices_dict=train_prices,
        symbols=symbols,
        initial_capital=trading_config["initial_capital"],
        leverage=trading_config["leverage"],
        maker_fee=trading_config["fees"]["maker"],
        taker_fee=trading_config["fees"]["taker"],
        slippage=trading_config["slippage"],
        max_position_per_asset=trading_config["max_position_per_asset"],
        episode_length=env_config["episode_length"],
        reward_scaling=env_config["reward_scaling"],
        transaction_penalty=env_config["transaction_penalty"],
        drawdown_penalty=env_config["drawdown_penalty"],
    )

    # Evaluation environment
    eval_env = MultiAssetTradingEnv(
        features_dict=eval_features,
        prices_dict=eval_prices,
        symbols=symbols,
        initial_capital=trading_config["initial_capital"],
        leverage=trading_config["leverage"],
        maker_fee=trading_config["fees"]["maker"],
        taker_fee=trading_config["fees"]["taker"],
        slippage=trading_config["slippage"],
        max_position_per_asset=trading_config["max_position_per_asset"],
        episode_length=min(len(list(eval_features.values())[0]) - 1, env_config["episode_length"]),  # Bounded length
        reward_scaling=env_config["reward_scaling"],
        transaction_penalty=env_config["transaction_penalty"],
        drawdown_penalty=env_config["drawdown_penalty"],
    )

    return train_env, eval_env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL-Crypto PPO Agent")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--steps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--force-download", action="store_true", help="Force re-download data")
    parser.add_argument("--dry-run", action="store_true", help="Quick test run")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume training from")
    args = parser.parse_args()

    # Setup logging
    logger.add(
        "logs/training_{time}.log",
        rotation="100 MB",
        level="INFO",
    )

    logger.info("=" * 60)
    logger.info("RL-Crypto PPO Training")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)

    if args.dry_run:
        logger.info("DRY RUN MODE: Using minimal settings")
        config["training"]["total_timesteps"] = args.steps or 1000
        config["training"]["eval_freq"] = 500
        config["data"]["history_days"] = 7

    # Prepare data
    train_features, train_prices, eval_features, eval_prices = prepare_data(
        config, force_download=args.force_download
    )

    if not train_features:
        logger.error("No data available for training!")
        return

    # Create environments
    train_env, eval_env = create_environments(
        config, train_features, train_prices, eval_features, eval_prices
    )

    logger.info(f"Training env: {train_env.n_assets} assets, {train_env.n_steps} steps")
    logger.info(f"Eval env: {eval_env.n_assets} assets, {eval_env.n_steps} steps")

    # Create agent
    agent = PPOAgent(train_env, config_path=args.config, eval_env=eval_env)

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        agent.load(args.resume)

    # Additional callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=config["training"].get("checkpoint_freq", 50000),
            save_path="models/checkpoints",
            verbose=1,
        ),
    ]

    if not args.dry_run:
        callbacks.append(
            EarlyStoppingCallback(
                check_freq=config["training"]["eval_freq"],
                patience=50,  # Increased patience
                min_episodes=200,
                verbose=1,
            )
        )

    # Train
    total_steps = args.steps or config["training"]["total_timesteps"]
    agent.train(total_timesteps=total_steps, callback_list=callbacks)

    # Save final model
    agent.save("models/ppo_trading_final")

    # Final evaluation
    logger.info("Running final evaluation...")
    metrics = agent.evaluate(n_episodes=10)

    logger.info("=" * 60)
    logger.info("Final Evaluation Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
