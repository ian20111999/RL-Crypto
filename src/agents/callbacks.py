"""Custom callbacks for PPO training."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback as SB3EvalCallback


class TradingCallback(BaseCallback):
    """Callback for logging trading-specific metrics during training."""

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 0,
    ):
        """Initialize callback.

        Args:
            log_freq: Frequency of logging (in steps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq

        # Tracking
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_returns: list[float] = []
        self.episode_drawdowns: list[float] = []
        self.current_rewards: list[float] = []

    def _on_step(self) -> bool:
        """Called after each step.

        Returns:
            Whether training should continue
        """
        # Track rewards
        if len(self.locals.get("rewards", [])) > 0:
            self.current_rewards.append(self.locals["rewards"][0])

        # Check for episode end
        if self.locals.get("dones", [False])[0]:
            # Get episode metrics from info
            infos = self.locals.get("infos", [{}])
            if infos:
                info = infos[0]

                episode_reward = sum(self.current_rewards)
                episode_length = len(self.current_rewards)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                if "total_return" in info:
                    self.episode_returns.append(info["total_return"])
                if "max_drawdown" in info:
                    self.episode_drawdowns.append(info["max_drawdown"])

                # Reset current episode tracking
                self.current_rewards = []

        # Log periodically
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Log aggregated metrics."""
        # Recent episodes
        recent_n = min(100, len(self.episode_rewards))

        metrics = {
            "trading/mean_episode_reward": np.mean(self.episode_rewards[-recent_n:]),
            "trading/mean_episode_length": np.mean(self.episode_lengths[-recent_n:]),
            "trading/total_episodes": len(self.episode_rewards),
        }

        if self.episode_returns:
            metrics["trading/mean_return"] = np.mean(self.episode_returns[-recent_n:])
            metrics["trading/std_return"] = np.std(self.episode_returns[-recent_n:])

        if self.episode_drawdowns:
            metrics["trading/mean_max_drawdown"] = np.mean(self.episode_drawdowns[-recent_n:])

        # Log to tensorboard
        for key, value in metrics.items():
            self.logger.record(key, value)

        if self.verbose > 0:
            logger.info(
                f"Step {self.n_calls}: "
                f"Episodes={len(self.episode_rewards)}, "
                f"MeanReward={metrics['trading/mean_episode_reward']:.2f}"
            )

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if len(self.episode_rewards) > 0:
            logger.info("=" * 50)
            logger.info("Training Summary:")
            logger.info(f"  Total Episodes: {len(self.episode_rewards)}")
            logger.info(f"  Mean Reward: {np.mean(self.episode_rewards):.2f}")
            logger.info(f"  Std Reward: {np.std(self.episode_rewards):.2f}")
            if self.episode_returns:
                logger.info(f"  Mean Return: {np.mean(self.episode_returns)*100:.2f}%")
            if self.episode_drawdowns:
                logger.info(f"  Mean Max DD: {np.mean(self.episode_drawdowns)*100:.2f}%")
            logger.info("=" * 50)


class EvalCallback(SB3EvalCallback):
    """Extended evaluation callback with trading-specific metrics."""

    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        """Initialize callback.

        Args:
            eval_env: Evaluation environment
            n_eval_episodes: Number of episodes per evaluation
            eval_freq: Evaluation frequency (in steps)
            best_model_save_path: Path to save best model
            deterministic: Whether to use deterministic policy
            verbose: Verbosity level
        """
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose,
        )

        self.eval_returns: list[float] = []
        self.eval_sharpe: list[float] = []
        self.eval_drawdowns: list[float] = []
        self.best_sharpe = -np.inf

    def _on_step(self) -> bool:
        """Called after each step."""
        result = super()._on_step()

        # After evaluation, extract trading metrics
        if self.n_calls % self.eval_freq == 0:
            self._extract_trading_metrics()

        return result

    def _extract_trading_metrics(self) -> None:
        """Extract and log trading-specific metrics from evaluation."""
        try:
            # Get base environment
            base_env = self.eval_env.envs[0]
            if hasattr(base_env, "env"):
                base_env = base_env.env

            if hasattr(base_env, "get_performance_metrics"):
                metrics = base_env.get_performance_metrics()

                if "total_return" in metrics:
                    self.eval_returns.append(metrics["total_return"])
                    self.logger.record("eval/total_return", metrics["total_return"])

                if "sharpe_ratio" in metrics:
                    self.eval_sharpe.append(metrics["sharpe_ratio"])
                    self.logger.record("eval/sharpe_ratio", metrics["sharpe_ratio"])

                    # Save best by Sharpe ratio
                    if metrics["sharpe_ratio"] > self.best_sharpe:
                        self.best_sharpe = metrics["sharpe_ratio"]
                        if self.best_model_save_path is not None:
                            path = Path(self.best_model_save_path) / "best_sharpe_model"
                            self.model.save(str(path))
                            logger.info(f"New best Sharpe: {self.best_sharpe:.2f}, saved to {path}")

                if "max_drawdown" in metrics:
                    self.eval_drawdowns.append(metrics["max_drawdown"])
                    self.logger.record("eval/max_drawdown", metrics["max_drawdown"])

                if "win_rate" in metrics:
                    self.logger.record("eval/win_rate", metrics["win_rate"])

                if "final_value" in metrics:
                    self.logger.record("eval/final_value", metrics["final_value"])

        except Exception as e:
            logger.warning(f"Failed to extract trading metrics: {e}")


class CheckpointCallback(BaseCallback):
    """Callback for periodic model checkpointing."""

    def __init__(
        self,
        save_freq: int = 50000,
        save_path: str = "models/checkpoints",
        name_prefix: str = "ppo",
        verbose: int = 0,
    ):
        """Initialize callback.

        Args:
            save_freq: Save frequency (in steps)
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint names
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls % self.save_freq == 0:
            path = self.save_path / f"{self.name_prefix}_{self.n_calls}_steps"
            self.model.save(str(path))
            if self.verbose > 0:
                logger.info(f"Checkpoint saved: {path}")

        return True


class EarlyStoppingCallback(BaseCallback):
    """Callback for early stopping based on performance metrics."""

    def __init__(
        self,
        reward_threshold: float = 0.0,
        check_freq: int = 10000,
        patience: int = 5,
        min_episodes: int = 100,
        verbose: int = 0,
    ):
        """Initialize callback.

        Args:
            reward_threshold: Minimum improvement threshold
            check_freq: Frequency of checks (in steps)
            patience: Number of checks without improvement before stopping
            min_episodes: Minimum episodes before early stopping can trigger
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.patience = patience
        self.min_episodes = min_episodes

        self.best_mean_reward = -np.inf
        self.checks_without_improvement = 0
        self.episode_rewards: list[float] = []
        self.current_rewards: list[float] = []

    def _on_step(self) -> bool:
        """Called after each step."""
        # Track rewards
        if len(self.locals.get("rewards", [])) > 0:
            self.current_rewards.append(self.locals["rewards"][0])

        if self.locals.get("dones", [False])[0]:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []

        # Check for improvement
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) >= self.min_episodes:
            mean_reward = np.mean(self.episode_rewards[-100:])

            if mean_reward > self.best_mean_reward + self.reward_threshold:
                self.best_mean_reward = mean_reward
                self.checks_without_improvement = 0
                if self.verbose > 0:
                    logger.info(f"New best mean reward: {mean_reward:.2f}")
            else:
                self.checks_without_improvement += 1
                if self.verbose > 0:
                    logger.info(
                        f"No improvement for {self.checks_without_improvement}/{self.patience} checks"
                    )

            if self.checks_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered at step {self.n_calls}")
                return False

        return True
