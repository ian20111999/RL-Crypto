"""Tests for trading environments."""

import numpy as np
import pytest

from src.envs import TradingEnv, MultiAssetTradingEnv


class TestTradingEnv:
    """Test suite for TradingEnv."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_steps = 1000
        lookback = 60
        n_features = 20

        # Generate random features
        features = np.random.randn(n_steps, lookback, n_features).astype(np.float32)

        # Generate realistic prices
        prices = np.zeros((n_steps, 4))
        price = 100.0
        for i in range(n_steps):
            change = np.random.randn() * 0.01
            price *= (1 + change)
            high = price * (1 + abs(np.random.randn() * 0.005))
            low = price * (1 - abs(np.random.randn() * 0.005))
            open_price = price * (1 + np.random.randn() * 0.002)
            prices[i] = [open_price, high, low, price]

        return features, prices

    def test_env_creation(self, sample_data):
        """Test environment creation."""
        features, prices = sample_data
        env = TradingEnv(features, prices)

        assert env.n_steps == len(features)
        assert env.initial_capital == 100.0

    def test_env_reset(self, sample_data):
        """Test environment reset."""
        features, prices = sample_data
        env = TradingEnv(features, prices)

        obs, info = env.reset()

        assert "market" in obs
        assert "portfolio" in obs
        assert obs["market"].shape == (60, 20)
        assert obs["portfolio"].shape == (4,)
        assert info["portfolio_value"] == 100.0

    def test_env_step(self, sample_data):
        """Test environment step."""
        features, prices = sample_data
        env = TradingEnv(features, prices)
        env.reset()

        action = np.array([0.5])  # 50% long
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "portfolio_value" in info

    def test_env_long_position(self, sample_data):
        """Test long position."""
        features, prices = sample_data
        env = TradingEnv(features, prices)
        env.reset()

        # Take long position
        action = np.array([1.0])
        env.step(action)

        assert env.position > 0

    def test_env_short_position(self, sample_data):
        """Test short position."""
        features, prices = sample_data
        env = TradingEnv(features, prices)
        env.reset()

        # Take short position
        action = np.array([-1.0])
        env.step(action)

        assert env.position < 0

    def test_env_episode(self, sample_data):
        """Test running a full episode."""
        features, prices = sample_data
        env = TradingEnv(features, prices, episode_length=100)
        env.reset()

        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps > 0
        assert "total_return" in info

    def test_performance_metrics(self, sample_data):
        """Test performance metrics calculation."""
        features, prices = sample_data
        env = TradingEnv(features, prices, episode_length=100)
        env.reset()

        # Run episode
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        metrics = env.get_performance_metrics()

        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics


class TestMultiAssetTradingEnv:
    """Test suite for MultiAssetTradingEnv."""

    @pytest.fixture
    def sample_multi_data(self):
        """Create sample multi-asset data for testing."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        n_steps = 500
        lookback = 60
        n_features = 20

        features_dict = {}
        prices_dict = {}

        for symbol in symbols:
            features_dict[symbol] = np.random.randn(n_steps, lookback, n_features).astype(np.float32)

            prices = np.zeros((n_steps, 4))
            price = 100.0
            for i in range(n_steps):
                change = np.random.randn() * 0.01
                price *= (1 + change)
                prices[i] = [price * 0.999, price * 1.002, price * 0.998, price]
            prices_dict[symbol] = prices

        return features_dict, prices_dict, symbols

    def test_multi_env_creation(self, sample_multi_data):
        """Test multi-asset environment creation."""
        features, prices, symbols = sample_multi_data
        env = MultiAssetTradingEnv(features, prices, symbols)

        assert env.n_assets == 3
        assert env.n_steps == 500

    def test_multi_env_reset(self, sample_multi_data):
        """Test multi-asset environment reset."""
        features, prices, symbols = sample_multi_data
        env = MultiAssetTradingEnv(features, prices, symbols)

        obs, info = env.reset()

        assert obs["market"].shape == (3, 60, 20)
        assert obs["portfolio"].shape == (5,)  # 3 positions + cash + value

    def test_multi_env_step(self, sample_multi_data):
        """Test multi-asset environment step."""
        features, prices, symbols = sample_multi_data
        env = MultiAssetTradingEnv(features, prices, symbols)
        env.reset()

        action = np.array([0.5, -0.3, 0.2])  # Different positions per asset
        obs, reward, terminated, truncated, info = env.step(action)

        assert len(info["positions"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
