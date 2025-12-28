"""Multi-agent ensemble for trading with voting mechanism."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml
from loguru import logger
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs import TradingEnv, MultiAssetTradingEnv


class MultiAgentEnsemble:
    """Ensemble of multiple RL agents with voting mechanism.
    
    Combines predictions from PPO, SAC, and TD3 using configurable
    voting strategies (average, weighted, majority).
    """
    
    SUPPORTED_ALGOS = {
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
    }
    
    def __init__(
        self,
        env: Union[TradingEnv, MultiAssetTradingEnv],
        config_path: str = "config/config.yaml",
        voting_strategy: str = "weighted_average",
    ):
        """Initialize multi-agent ensemble.
        
        Args:
            env: Trading environment
            config_path: Path to configuration file
            voting_strategy: Voting strategy ('average', 'weighted_average', 'majority')
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.env = env
        self.voting_strategy = voting_strategy
        
        # Agent storage
        self.agents: dict[str, object] = {}
        self.vec_envs: dict[str, VecNormalize] = {}
        self.weights: dict[str, float] = {}
        
        # Performance tracking for adaptive weights
        self.agent_returns: dict[str, list] = {}
        self.agent_sharpes: dict[str, list] = {}
        
        logger.info(f"MultiAgentEnsemble initialized with strategy: {voting_strategy}")

    def add_agent(
        self,
        name: str,
        algo: str,
        model_path: Optional[str] = None,
        weight: float = 1.0,
        **kwargs,
    ) -> None:
        """Add an agent to the ensemble.
        
        Args:
            name: Unique name for the agent
            algo: Algorithm type ('ppo', 'sac', 'td3')
            model_path: Path to pre-trained model (optional)
            weight: Initial weight for voting
            **kwargs: Additional arguments for model initialization
        """
        algo = algo.lower()
        if algo not in self.SUPPORTED_ALGOS:
            raise ValueError(f"Unsupported algorithm: {algo}. Use: {list(self.SUPPORTED_ALGOS.keys())}")
        
        AlgoClass = self.SUPPORTED_ALGOS[algo]
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )
        
        if model_path and Path(model_path).exists():
            # Load pre-trained model
            agent = AlgoClass.load(model_path, env=vec_env)
            
            # Load VecNormalize stats if available
            vec_norm_path = f"{model_path}_vecnormalize.pkl"
            if Path(vec_norm_path).exists():
                vec_env = VecNormalize.load(vec_norm_path, vec_env)
            
            logger.info(f"Loaded {algo.upper()} agent '{name}' from {model_path}")
        else:
            # Initialize new model
            default_kwargs = self._get_default_kwargs(algo)
            default_kwargs.update(kwargs)
            
            agent = AlgoClass(
                policy="MultiInputPolicy",
                env=vec_env,
                verbose=0,
                device="auto",
                **default_kwargs,
            )
            logger.info(f"Created new {algo.upper()} agent '{name}'")
        
        self.agents[name] = agent
        self.vec_envs[name] = vec_env
        self.weights[name] = weight
        self.agent_returns[name] = []
        self.agent_sharpes[name] = []

    def _get_default_kwargs(self, algo: str) -> dict:
        """Get default hyperparameters for an algorithm.
        
        Args:
            algo: Algorithm name
            
        Returns:
            Dict of default parameters
        """
        ppo_config = self.config.get("ppo", {})
        
        if algo == "ppo":
            return {
                "learning_rate": ppo_config.get("learning_rate", 3e-4),
                "n_steps": ppo_config.get("n_steps", 2048),
                "batch_size": ppo_config.get("batch_size", 64),
                "n_epochs": ppo_config.get("n_epochs", 10),
                "gamma": ppo_config.get("gamma", 0.99),
                "gae_lambda": ppo_config.get("gae_lambda", 0.95),
                "clip_range": ppo_config.get("clip_range", 0.2),
            }
        elif algo == "sac":
            return {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "batch_size": 256,
                "gamma": 0.99,
                "tau": 0.005,
                "learning_starts": 1000,
            }
        elif algo == "td3":
            return {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "batch_size": 256,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_delay": 2,
            }
        return {}

    def train_all(
        self,
        total_timesteps: int = 100000,
        progress_bar: bool = True,
    ) -> None:
        """Train all agents in the ensemble.
        
        Args:
            total_timesteps: Number of timesteps per agent
            progress_bar: Whether to show progress bar
        """
        for name, agent in self.agents.items():
            logger.info(f"Training agent '{name}'...")
            agent.learn(
                total_timesteps=total_timesteps,
                progress_bar=progress_bar,
            )
            logger.info(f"Agent '{name}' training complete")

    def predict(
        self,
        observation: dict,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Get ensemble prediction using voting.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic predictions
            
        Returns:
            Tuple of (action, info)
        """
        if not self.agents:
            raise ValueError("No agents in ensemble. Add agents first.")
        
        predictions = {}
        
        for name, agent in self.agents.items():
            action, _ = agent.predict(observation, deterministic=deterministic)
            predictions[name] = action
        
        # Apply voting strategy
        if self.voting_strategy == "average":
            ensemble_action = self._average_vote(predictions)
        elif self.voting_strategy == "weighted_average":
            ensemble_action = self._weighted_average_vote(predictions)
        elif self.voting_strategy == "majority":
            ensemble_action = self._majority_vote(predictions)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        
        # Info about individual predictions
        info = {
            "individual_predictions": predictions,
            "weights": self.weights.copy(),
        }
        
        return ensemble_action, info

    def _average_vote(self, predictions: dict) -> np.ndarray:
        """Simple average of all predictions.
        
        Args:
            predictions: Dict of agent predictions
            
        Returns:
            Averaged action
        """
        actions = list(predictions.values())
        return np.mean(actions, axis=0)

    def _weighted_average_vote(self, predictions: dict) -> np.ndarray:
        """Weighted average based on agent weights.
        
        Args:
            predictions: Dict of agent predictions
            
        Returns:
            Weighted averaged action
        """
        total_weight = sum(self.weights.values())
        weighted_sum = np.zeros_like(list(predictions.values())[0])
        
        for name, action in predictions.items():
            weighted_sum += action * self.weights[name]
        
        return weighted_sum / total_weight

    def _majority_vote(self, predictions: dict) -> np.ndarray:
        """Majority vote (for discrete action components).
        
        For continuous actions, discretizes into buy/sell/hold.
        
        Args:
            predictions: Dict of agent predictions
            
        Returns:
            Majority voted action
        """
        # Discretize actions: -1 = sell, 0 = hold, 1 = buy
        discrete = {}
        for name, action in predictions.items():
            discrete[name] = np.sign(action)
        
        # Vote
        votes = np.stack(list(discrete.values()))
        majority = np.sign(np.sum(votes, axis=0))
        
        # Use average magnitude from agents that agree
        magnitudes = []
        for name, action in predictions.items():
            if np.all(np.sign(action) == majority):
                magnitudes.append(np.abs(action))
        
        if magnitudes:
            avg_magnitude = np.mean(magnitudes, axis=0)
        else:
            avg_magnitude = np.mean(np.abs(list(predictions.values())), axis=0)
        
        return majority * avg_magnitude

    def update_weights(self, agent_name: str, return_val: float, sharpe: float) -> None:
        """Update agent weight based on recent performance.
        
        Args:
            agent_name: Name of the agent
            return_val: Recent return
            sharpe: Recent Sharpe ratio
        """
        if agent_name not in self.agents:
            return
        
        self.agent_returns[agent_name].append(return_val)
        self.agent_sharpes[agent_name].append(sharpe)
        
        # Keep only last 100 values
        self.agent_returns[agent_name] = self.agent_returns[agent_name][-100:]
        self.agent_sharpes[agent_name] = self.agent_sharpes[agent_name][-100:]
        
        # Update weights based on Sharpe (softmax)
        if len(self.agent_sharpes[agent_name]) >= 10:
            self._recompute_weights()

    def _recompute_weights(self) -> None:
        """Recompute weights based on recent performance."""
        avg_sharpes = {}
        for name in self.agents:
            if self.agent_sharpes[name]:
                avg_sharpes[name] = np.mean(self.agent_sharpes[name][-20:])
            else:
                avg_sharpes[name] = 0.0
        
        # Softmax
        sharpe_values = np.array(list(avg_sharpes.values()))
        exp_sharpes = np.exp(sharpe_values - np.max(sharpe_values))  # Numerical stability
        softmax_weights = exp_sharpes / exp_sharpes.sum()
        
        for i, name in enumerate(avg_sharpes.keys()):
            self.weights[name] = softmax_weights[i]
        
        logger.debug(f"Updated ensemble weights: {self.weights}")

    def save_all(self, base_path: str = "models/ensemble") -> None:
        """Save all models in the ensemble.
        
        Args:
            base_path: Base directory to save models
        """
        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)
        
        for name, agent in self.agents.items():
            model_path = base / f"{name}"
            agent.save(str(model_path))
            
            # Save VecNormalize stats
            if name in self.vec_envs:
                self.vec_envs[name].save(f"{model_path}_vecnormalize.pkl")
        
        # Save weights
        import json
        with open(base / "weights.json", "w") as f:
            json.dump(self.weights, f)
        
        logger.info(f"Ensemble saved to {base_path}")

    def load_all(self, base_path: str = "models/ensemble") -> None:
        """Load all models from a saved ensemble.
        
        Args:
            base_path: Base directory with saved models
        """
        base = Path(base_path)
        
        # Load weights
        weights_path = base / "weights.json"
        if weights_path.exists():
            import json
            with open(weights_path) as f:
                saved_weights = json.load(f)
        else:
            saved_weights = {}
        
        # Find and load models
        for model_file in base.glob("*.zip"):
            name = model_file.stem
            
            # Determine algorithm from model
            try:
                # Try each algorithm
                for algo, AlgoClass in self.SUPPORTED_ALGOS.items():
                    try:
                        self.add_agent(
                            name=name,
                            algo=algo,
                            model_path=str(model_file),
                            weight=saved_weights.get(name, 1.0),
                        )
                        break
                    except:
                        continue
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")
        
        logger.info(f"Loaded ensemble from {base_path}")

    def get_performance_summary(self) -> dict:
        """Get summary of ensemble performance.
        
        Returns:
            Dict with performance metrics
        """
        summary = {
            "n_agents": len(self.agents),
            "agents": list(self.agents.keys()),
            "voting_strategy": self.voting_strategy,
            "weights": self.weights.copy(),
        }
        
        for name in self.agents:
            if self.agent_returns[name]:
                summary[f"{name}_avg_return"] = np.mean(self.agent_returns[name])
            if self.agent_sharpes[name]:
                summary[f"{name}_avg_sharpe"] = np.mean(self.agent_sharpes[name])
        
        return summary
