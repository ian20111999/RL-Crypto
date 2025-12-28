"""Parallel training environment using SubprocVecEnv for multi-CPU training."""

from typing import Callable, Optional
import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from loguru import logger

from src.envs import MultiAssetTradingEnv


def make_env(
    features_dict: dict,
    prices_dict: dict,
    symbols: list[str],
    config: dict,
    seed: int = 0,
) -> Callable:
    """Create environment factory function.
    
    Args:
        features_dict: Features for each symbol
        prices_dict: Prices for each symbol
        symbols: List of trading symbols
        config: Configuration dict
        seed: Random seed
        
    Returns:
        Callable that creates environment
    """
    def _init() -> MultiAssetTradingEnv:
        env = MultiAssetTradingEnv(
            features_dict=features_dict,
            prices_dict=prices_dict,
            symbols=symbols,
            initial_capital=config["trading"]["initial_capital"],
            leverage=config["trading"]["leverage"],
            maker_fee=config["trading"]["fees"]["maker"],
            taker_fee=config["trading"]["fees"]["taker"],
            slippage=config["trading"]["slippage"],
            max_position_per_asset=config["trading"]["max_position_per_asset"],
            episode_length=config["environment"]["episode_length"],
            reward_scaling=config["environment"]["reward_scaling"],
            transaction_penalty=config["environment"]["transaction_penalty"],
            drawdown_penalty=config["environment"]["drawdown_penalty"],
        )
        env.reset(seed=seed)
        return env
    
    return _init


def create_parallel_envs(
    features_dict: dict,
    prices_dict: dict,
    symbols: list[str],
    config: dict,
    n_envs: Optional[int] = None,
    use_subprocess: bool = True,
    normalize: bool = True,
) -> VecNormalize:
    """Create parallel training environments.
    
    Args:
        features_dict: Features for each symbol
        prices_dict: Prices for each symbol
        symbols: List of trading symbols
        config: Configuration dict
        n_envs: Number of parallel environments (default: CPU count)
        use_subprocess: Use SubprocVecEnv (True) or DummyVecEnv (False)
        normalize: Apply VecNormalize wrapper
        
    Returns:
        Vectorized environment
    """
    if n_envs is None:
        n_envs = min(mp.cpu_count(), 8)  # Cap at 8
    
    logger.info(f"Creating {n_envs} parallel environments (subprocess={use_subprocess})")
    
    env_fns = [
        make_env(features_dict, prices_dict, symbols, config, seed=i)
        for i in range(n_envs)
    ]
    
    if use_subprocess and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="fork")
    else:
        vec_env = DummyVecEnv(env_fns)
    
    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
    
    return vec_env


class AsyncDataLoader:
    """Async data loader for parallel training."""
    
    def __init__(
        self,
        data_queue_size: int = 10,
        prefetch: int = 2,
    ):
        """Initialize async data loader.
        
        Args:
            data_queue_size: Size of data queue
            prefetch: Number of batches to prefetch
        """
        self.queue_size = data_queue_size
        self.prefetch = prefetch
        self._queue = None
        self._workers = []
        self._running = False

    def start(self, data_generator: Callable) -> None:
        """Start data loading workers.
        
        Args:
            data_generator: Callable that yields data batches
        """
        import queue
        import threading
        
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._running = True
        
        def worker():
            while self._running:
                try:
                    batch = next(data_generator())
                    self._queue.put(batch, timeout=1)
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Data loader error: {e}")
        
        for _ in range(self.prefetch):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            self._workers.append(t)

    def get_batch(self, timeout: float = 5.0):
        """Get next batch from queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Data batch
        """
        return self._queue.get(timeout=timeout)

    def stop(self) -> None:
        """Stop data loading workers."""
        self._running = False
        for worker in self._workers:
            worker.join(timeout=1)
        self._workers.clear()


def get_optimal_n_envs(
    features_dict: dict,
    target_memory_gb: float = 4.0,
) -> int:
    """Calculate optimal number of parallel environments.
    
    Args:
        features_dict: Features dict
        target_memory_gb: Target memory usage in GB
        
    Returns:
        Recommended number of environments
    """
    # Estimate memory per env
    sample_features = list(features_dict.values())[0]
    memory_per_env_mb = sample_features.nbytes / 1024 / 1024 * len(features_dict)
    
    # Add overhead
    memory_per_env_mb *= 2  # Approximate overhead
    
    # Calculate based on target
    n_envs = int(target_memory_gb * 1024 / memory_per_env_mb)
    n_envs = max(1, min(n_envs, mp.cpu_count()))
    
    logger.info(
        f"Recommended n_envs: {n_envs} "
        f"(~{memory_per_env_mb:.0f}MB per env, target {target_memory_gb}GB)"
    )
    
    return n_envs
