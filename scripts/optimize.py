"""Hyperparameter optimization with Optuna for PPO trading agent."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import optuna
import yaml
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import torch

from src.data import DataCollector, DataProcessor
from src.envs import MultiAssetTradingEnv


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> tuple:
    """Prepare training and evaluation data."""
    collector = DataCollector("config/config.yaml")
    processor = DataProcessor(lookback=config["data"]["lookback"])

    raw_data = collector.load_all()
    processed_data = processor.process_multiple(raw_data, fit=True)

    symbols = config["symbols"]
    train_features_dict = {}
    train_prices_dict = {}
    eval_features_dict = {}
    eval_prices_dict = {}

    train_ratio = 0.8

    for symbol in symbols:
        if symbol not in processed_data or processed_data[symbol].empty:
            continue

        df = processed_data[symbol]
        features, prices, _ = processor.create_sequences(df)

        split_idx = int(len(features) * train_ratio)

        train_features_dict[symbol] = features[:split_idx]
        train_prices_dict[symbol] = prices[:split_idx]
        eval_features_dict[symbol] = features[split_idx:]
        eval_prices_dict[symbol] = prices[split_idx:]

    return train_features_dict, train_prices_dict, eval_features_dict, eval_prices_dict


def create_env(
    config: dict,
    features_dict: dict,
    prices_dict: dict,
    episode_length: int = 1440,
) -> MultiAssetTradingEnv:
    """Create trading environment."""
    symbols = list(features_dict.keys())
    trading_config = config["trading"]
    env_config = config["environment"]

    return MultiAssetTradingEnv(
        features_dict=features_dict,
        prices_dict=prices_dict,
        symbols=symbols,
        initial_capital=trading_config["initial_capital"],
        leverage=trading_config["leverage"],
        maker_fee=trading_config["fees"]["maker"],
        taker_fee=trading_config["fees"]["taker"],
        slippage=trading_config["slippage"],
        max_position_per_asset=trading_config["max_position_per_asset"],
        episode_length=episode_length,
        reward_scaling=env_config["reward_scaling"],
        transaction_penalty=env_config["transaction_penalty"],
        drawdown_penalty=env_config["drawdown_penalty"],
    )


def sample_ppo_params(trial: optuna.Trial) -> dict:
    """Sample PPO hyperparameters for optimization.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dict of sampled hyperparameters
    """
    # Learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    # PPO specific
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    
    # Entropy and value function
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    
    # Network architecture
    net_arch_type = trial.suggest_categorical("net_arch_type", ["small", "medium", "large"])
    
    net_arch_map = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[128, 128], vf=[128, 128]),
        "large": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    }
    
    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch_map[net_arch_type],
            "activation_fn": torch.nn.ReLU,
        },
    }


def sample_reward_params(trial: optuna.Trial) -> dict:
    """Sample reward function hyperparameters.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dict of reward parameters
    """
    return {
        "reward_scaling": trial.suggest_float("reward_scaling", 10, 500, log=True),
        "transaction_penalty": trial.suggest_float("transaction_penalty", 0.0001, 0.01, log=True),
        "drawdown_penalty": trial.suggest_float("drawdown_penalty", 0.01, 0.5),
    }


class OptunaObjective:
    """Optuna objective for PPO hyperparameter optimization."""
    
    def __init__(
        self,
        config: dict,
        train_features: dict,
        train_prices: dict,
        eval_features: dict,
        eval_prices: dict,
        n_timesteps: int = 50000,
        n_eval_episodes: int = 5,
        optimize_reward: bool = False,
    ):
        """Initialize objective.
        
        Args:
            config: Configuration dict
            train_features: Training features dict
            train_prices: Training prices dict
            eval_features: Evaluation features dict
            eval_prices: Evaluation prices dict
            n_timesteps: Number of training timesteps per trial
            n_eval_episodes: Number of episodes for evaluation
            optimize_reward: Whether to also optimize reward function params
        """
        self.config = config
        self.train_features = train_features
        self.train_prices = train_prices
        self.eval_features = eval_features
        self.eval_prices = eval_prices
        self.n_timesteps = n_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.optimize_reward = optimize_reward
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate a trial.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Objective value (negative Sharpe ratio for minimization)
        """
        # Sample hyperparameters
        ppo_params = sample_ppo_params(trial)
        
        # Optionally sample reward params
        if self.optimize_reward:
            reward_params = sample_reward_params(trial)
        else:
            reward_params = {
                "reward_scaling": self.config["environment"]["reward_scaling"],
                "transaction_penalty": self.config["environment"]["transaction_penalty"],
                "drawdown_penalty": self.config["environment"]["drawdown_penalty"],
            }
        
        try:
            # Create environments with possibly modified reward params
            config_copy = self.config.copy()
            config_copy["environment"] = {**config_copy["environment"], **reward_params}
            
            train_env = create_env(
                config_copy,
                self.train_features,
                self.train_prices,
            )
            eval_env = create_env(
                config_copy,
                self.eval_features,
                self.eval_prices,
                episode_length=None,
            )
            
            # Wrap environments
            train_vec_env = DummyVecEnv([lambda: train_env])
            train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
            
            eval_vec_env = DummyVecEnv([lambda: eval_env])
            eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, training=False)
            
            # Create model
            model = PPO(
                policy="MultiInputPolicy",
                env=train_vec_env,
                **ppo_params,
                verbose=0,
                device="auto",
            )
            
            # Train
            model.learn(
                total_timesteps=self.n_timesteps,
                progress_bar=False,
            )
            
            # Evaluate
            returns = []
            sharpe_ratios = []
            
            for _ in range(self.n_eval_episodes):
                obs = eval_vec_env.reset()
                done = False
                episode_returns = []
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, dones, info = eval_vec_env.step(action)
                    episode_returns.append(reward[0])
                    done = dones[0]
                
                # Get metrics from environment
                base_env = eval_vec_env.envs[0]
                if hasattr(base_env, "get_performance_metrics"):
                    metrics = base_env.get_performance_metrics()
                    if metrics:
                        returns.append(metrics.get("total_return", 0))
                        sharpe_ratios.append(metrics.get("sharpe_ratio", 0))
            
            # Calculate objective (maximize Sharpe ratio)
            if sharpe_ratios:
                mean_sharpe = np.mean(sharpe_ratios)
                mean_return = np.mean(returns)
                
                # Report intermediate values
                trial.report(mean_sharpe, step=0)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                logger.info(
                    f"Trial {trial.number}: "
                    f"Sharpe={mean_sharpe:.2f}, Return={mean_return*100:.2f}%"
                )
                
                return mean_sharpe
            else:
                return float("-inf")
                
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float("-inf")


def run_optimization(
    config_path: str = "config/config.yaml",
    n_trials: int = 50,
    n_timesteps: int = 50000,
    optimize_reward: bool = False,
    study_name: str = "ppo_optimization",
    storage: str = None,
) -> optuna.Study:
    """Run hyperparameter optimization.
    
    Args:
        config_path: Path to config file
        n_trials: Number of optimization trials
        n_timesteps: Training timesteps per trial
        optimize_reward: Whether to optimize reward function
        study_name: Name of the Optuna study
        storage: Optional database URL for persistent storage
        
    Returns:
        Optuna study object
    """
    logger.info("=" * 60)
    logger.info("PPO Hyperparameter Optimization with Optuna")
    logger.info("=" * 60)
    
    config = load_config(config_path)
    
    # Prepare data
    logger.info("Preparing data...")
    train_features, train_prices, eval_features, eval_prices = prepare_data(config)
    
    if not train_features:
        raise ValueError("No data available!")
    
    # Create objective
    objective = OptunaObjective(
        config=config,
        train_features=train_features,
        train_prices=train_prices,
        eval_features=eval_features,
        eval_prices=eval_prices,
        n_timesteps=n_timesteps,
        optimize_reward=optimize_reward,
    )
    
    # Create or load study
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        )
    
    # Run optimization
    logger.info(f"Starting optimization with {n_trials} trials...")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,  # Sequential for GPU memory
    )
    
    # Report results
    logger.info("=" * 60)
    logger.info("Optimization Complete!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best Sharpe ratio: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    # Save best params to file
    best_params_path = Path("config") / "best_params.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(study.best_params, f, default_flow_style=False)
    logger.info(f"Best parameters saved to {best_params_path}")
    
    return study


def main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(description="Optimize PPO hyperparameters with Optuna")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps per trial")
    parser.add_argument("--optimize-reward", action="store_true", help="Also optimize reward function")
    parser.add_argument("--study-name", type=str, default="ppo_trading_optuna", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Database URL for persistent storage")
    parser.add_argument("--dashboard", action="store_true", help="Launch Optuna dashboard after optimization")
    args = parser.parse_args()
    
    # Setup logging
    logger.add(
        "logs/optuna_{time}.log",
        rotation="100 MB",
        level="INFO",
    )
    
    study = run_optimization(
        config_path=args.config,
        n_trials=args.trials,
        n_timesteps=args.timesteps,
        optimize_reward=args.optimize_reward,
        study_name=args.study_name,
        storage=args.storage,
    )
    
    # Optionally launch dashboard
    if args.dashboard and args.storage:
        import subprocess
        logger.info(f"Launching Optuna Dashboard at http://localhost:8080")
        subprocess.run(["optuna-dashboard", args.storage])


if __name__ == "__main__":
    main()
