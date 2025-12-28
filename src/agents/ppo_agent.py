"""PPO Agent wrapper for cryptocurrency trading."""

from pathlib import Path
from typing import Any, Optional, Union

import torch
import yaml
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs import TradingEnv, MultiAssetTradingEnv
from src.agents.callbacks import TradingCallback, EvalCallback


class PPOAgent:
    """PPO Agent wrapper with training and evaluation utilities."""

    def __init__(
        self,
        env: Union[TradingEnv, MultiAssetTradingEnv],
        config_path: str = "config/config.yaml",
        eval_env: Optional[Union[TradingEnv, MultiAssetTradingEnv]] = None,
    ):
        """Initialize PPO Agent.

        Args:
            env: Trading environment
            config_path: Path to configuration file
            eval_env: Optional separate environment for evaluation
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.ppo_config = self.config["ppo"]
        self.training_config = self.config["training"]

        # Setup directories
        self.model_dir = Path(self.training_config["model_dir"])
        self.log_dir = Path(self.training_config["log_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Wrap environment
        self.env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(
            self.env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        # Evaluation environment
        if eval_env is not None:
            self.eval_env = DummyVecEnv([lambda: eval_env])
            self.eval_env = VecNormalize(
                self.eval_env,
                norm_obs=True,
                norm_reward=False,
                training=False,
            )
        else:
            self.eval_env = None

        # Build policy kwargs
        policy_kwargs = self._build_policy_kwargs()

        # Initialize PPO model
        self.model = PPO(
            policy=self.ppo_config["policy"],
            env=self.env,
            learning_rate=self.ppo_config["learning_rate"],
            n_steps=self.ppo_config["n_steps"],
            batch_size=self.ppo_config["batch_size"],
            n_epochs=self.ppo_config["n_epochs"],
            gamma=self.ppo_config["gamma"],
            gae_lambda=self.ppo_config["gae_lambda"],
            clip_range=self.ppo_config["clip_range"],
            vf_coef=self.ppo_config["vf_coef"],
            ent_coef=self.ppo_config["ent_coef"],
            max_grad_norm=self.ppo_config["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(self.log_dir),
            seed=self.training_config["seed"],
            verbose=1,
            device="auto",
        )

        logger.info(f"PPO Agent initialized with device: {self.model.device}")

    def _build_policy_kwargs(self) -> dict:
        """Build policy kwargs from config.

        Returns:
            Policy kwargs dict
        """
        net_arch = self.ppo_config.get("policy_kwargs", {}).get("net_arch", {})
        activation_fn_name = self.ppo_config.get("policy_kwargs", {}).get(
            "activation_fn", "ReLU"
        )

        # Get activation function
        activation_fn = getattr(torch.nn, activation_fn_name)

        policy_kwargs = {
            "net_arch": dict(
                pi=net_arch.get("pi", [256, 256]),
                vf=net_arch.get("vf", [256, 256]),
            ),
            "activation_fn": activation_fn,
        }

        return policy_kwargs

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback_list: Optional[list] = None,
        progress_bar: bool = True,
    ) -> None:
        """Train the PPO agent.

        Args:
            total_timesteps: Total number of timesteps (uses config default if not specified)
            callback_list: List of additional callbacks
            progress_bar: Whether to show progress bar
        """
        total_timesteps = total_timesteps or self.training_config["total_timesteps"]

        # Setup callbacks
        callbacks = []

        # Trading metrics callback
        trading_callback = TradingCallback(
            log_freq=self.training_config.get("eval_freq", 10000) // 10,
            verbose=1,
        )
        callbacks.append(trading_callback)

        # Evaluation callback
        if self.eval_env is not None:
            eval_callback = EvalCallback(
                eval_env=self.eval_env,
                n_eval_episodes=self.training_config.get("n_eval_episodes", 5),
                eval_freq=self.training_config.get("eval_freq", 10000),
                best_model_save_path=str(self.model_dir / "best"),
                deterministic=True,
                verbose=1,
            )
            callbacks.append(eval_callback)

        # Add custom callbacks
        if callback_list:
            callbacks.extend(callback_list)

        callback = CallbackList(callbacks)

        logger.info(f"Starting training for {total_timesteps} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
            tb_log_name="ppo_trading",
        )

        logger.info("Training completed!")

    def predict(
        self,
        observation: dict,
        deterministic: bool = True,
    ) -> tuple:
        """Predict action for given observation.

        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, state)
        """
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state

    def evaluate(
        self,
        env: Optional[Union[TradingEnv, MultiAssetTradingEnv]] = None,
        n_episodes: int = 5,
        deterministic: bool = True,
    ) -> dict:
        """Evaluate the agent on an environment.

        Args:
            env: Environment to evaluate on (uses eval_env if not specified)
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy

        Returns:
            Dict of evaluation metrics
        """
        if env is None:
            if self.eval_env is not None:
                eval_vec_env = self.eval_env
            else:
                eval_vec_env = self.env
        else:
            eval_vec_env = DummyVecEnv([lambda: env])

        all_metrics = []
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs = eval_vec_env.reset()
            done = False
            total_reward = 0
            length = 0
            max_steps = 2000  # Prevent infinite loops

            while not done and length < max_steps:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, dones, info = eval_vec_env.step(action)
                total_reward += reward[0]
                length += 1
                done = dones[0]

            # Get environment metrics at end of episode
            if hasattr(eval_vec_env, "envs"):
                base_env = eval_vec_env.envs[0]
                if hasattr(base_env, "env"):
                    base_env = base_env.env
                if hasattr(base_env, "get_performance_metrics"):
                    metrics = base_env.get_performance_metrics()
                    all_metrics.append(metrics)

            episode_rewards.append(total_reward)
            episode_lengths.append(length)

        # Aggregate metrics
        import numpy as np

        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "n_episodes": n_episodes,
        }

        if all_metrics:
            for key in all_metrics[0].keys():
                if isinstance(all_metrics[0][key], (int, float)):
                    values = [m.get(key, 0) for m in all_metrics]
                    results[f"mean_{key}"] = np.mean(values)
                    results[f"std_{key}"] = np.std(values)

        logger.info(f"Evaluation Results: {results}")
        return results

    def save(self, path: Optional[str] = None) -> str:
        """Save the model.

        Args:
            path: Path to save model (uses default if not specified)

        Returns:
            Path where model was saved
        """
        if path is None:
            path = str(self.model_dir / "ppo_trading")

        self.model.save(path)

        # Save VecNormalize stats
        vec_norm_path = f"{path}_vecnormalize.pkl"
        self.env.save(vec_norm_path)

        logger.info(f"Model saved to {path}")
        return path

    def load(self, path: str) -> None:
        """Load a saved model.

        Args:
            path: Path to load model from
        """
        self.model = PPO.load(path, env=self.env)

        # Load VecNormalize stats if available
        vec_norm_path = f"{path}_vecnormalize.pkl"
        if Path(vec_norm_path).exists():
            self.env = VecNormalize.load(vec_norm_path, self.env)

        logger.info(f"Model loaded from {path}")

    def get_policy_weights(self) -> dict:
        """Get policy network weights.

        Returns:
            Dict of policy weights
        """
        return {
            name: param.data.cpu().numpy()
            for name, param in self.model.policy.named_parameters()
        }
