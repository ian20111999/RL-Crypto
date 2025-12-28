"""Multi-asset trading environment for RL training."""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger


class MultiAssetTradingEnv(gym.Env):
    """Gymnasium environment for multi-asset cryptocurrency futures trading.

    This environment simulates trading multiple cryptocurrency futures contracts
    simultaneously with continuous position sizing from -1 (full short) to +1 (full long)
    for each asset.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        features_dict: dict[str, np.ndarray],
        prices_dict: dict[str, np.ndarray],
        symbols: list[str],
        initial_capital: float = 100.0,
        leverage: int = 2,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        slippage: float = 0.0001,
        max_position_per_asset: float = 0.2,
        episode_length: Optional[int] = None,
        reward_scaling: float = 100.0,
        transaction_penalty: float = 0.001,
        drawdown_penalty: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        """Initialize multi-asset trading environment.

        Args:
            features_dict: Dict mapping symbol to feature array (n_steps, lookback, n_features)
            prices_dict: Dict mapping symbol to price array (n_steps, 4) for OHLC
            symbols: List of trading symbols
            initial_capital: Starting capital in USDT
            leverage: Leverage multiplier
            maker_fee: Maker fee rate
            taker_fee: Taker fee rate
            slippage: Slippage rate
            max_position_per_asset: Maximum position size per asset as fraction of capital
            episode_length: Maximum steps per episode
            reward_scaling: Reward multiplier
            transaction_penalty: Penalty for position changes
            drawdown_penalty: Penalty coefficient for drawdown
            render_mode: Rendering mode
        """
        super().__init__()

        self.symbols = symbols
        self.n_assets = len(symbols)

        # Store data
        self.features_dict = features_dict
        self.prices_dict = prices_dict

        # Get dimensions from first symbol
        first_symbol = symbols[0]
        self.n_steps = len(features_dict[first_symbol])
        self.lookback = features_dict[first_symbol].shape[1]
        self.n_features = features_dict[first_symbol].shape[2]

        # Trading parameters
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.max_position_per_asset = max_position_per_asset
        self.episode_length = episode_length or self.n_steps

        # Reward parameters
        self.reward_scaling = reward_scaling
        self.transaction_penalty = transaction_penalty
        self.drawdown_penalty = drawdown_penalty

        self.render_mode = render_mode

        # Observation space
        self.observation_space = spaces.Dict({
            # Market features for each asset
            "market": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_assets, self.lookback, self.n_features),
                dtype=np.float32,
            ),
            # Portfolio state: positions + cash_pct + total_value_pct
            "portfolio": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_assets + 2,),
                dtype=np.float32,
            ),
        })

        # Action space: continuous position for each asset
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # Initialize state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.current_step = 0
        self.cash = self.initial_capital
        self.positions = np.zeros(self.n_assets)  # Position for each asset
        self.entry_prices = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        self.prev_positions = np.zeros(self.n_assets)

        # Tracking
        self.returns_history: list[float] = []
        self.position_history: list[np.ndarray] = []
        self.value_history: list[float] = []
        self.trade_counts = np.zeros(self.n_assets, dtype=int)
        self.total_fees = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        self._reset_state()

        # Optional: start from a random point
        if options and "start_step" in options:
            self.current_step = options["start_step"]
        elif self.n_steps > self.episode_length:
            max_start = self.n_steps - self.episode_length
            self.current_step = self.np_random.integers(0, max_start)

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: Target positions array of shape (n_assets,)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Clip actions to valid range
        target_positions = np.clip(action, -1.0, 1.0)

        # Scale by max position per asset
        target_positions = target_positions * self.max_position_per_asset

        # Get current prices for all assets
        current_prices = self._get_current_prices()

        # Execute trades
        for i, symbol in enumerate(self.symbols):
            position_change = target_positions[i] - self.positions[i]
            if abs(position_change) > 0.01:
                self._execute_trade(i, position_change, current_prices[i])

        # Update portfolio value
        self._update_portfolio_value(current_prices)

        # Calculate reward
        reward = self._calculate_reward(target_positions)

        # Update tracking
        step_return = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        self.returns_history.append(step_return)
        self.position_history.append(self.positions.copy())
        self.value_history.append(self.portfolio_value)

        # Update previous values
        self.prev_portfolio_value = self.portfolio_value
        self.prev_positions = self.positions.copy()
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Advance step
        self.current_step += 1

        # Check termination
        terminated = False
        truncated = False

        if self.portfolio_value <= 0:
            terminated = True

        if self.peak_value > 0:
            current_dd = (self.peak_value - self.portfolio_value) / self.peak_value
            if current_dd > 0.5:
                terminated = True

        if self.current_step >= min(self.n_steps, self.episode_length):
            truncated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_current_prices(self) -> np.ndarray:
        """Get current close prices for all assets.

        Returns:
            Array of current prices
        """
        prices = np.zeros(self.n_assets)
        # Clamp to valid range
        step = min(self.current_step, self.n_steps - 1)
        for i, symbol in enumerate(self.symbols):
            prices[i] = self.prices_dict[symbol][step][3]  # Close price
        return prices

    def _execute_trade(self, asset_idx: int, position_change: float, price: float) -> None:
        """Execute a trade for a specific asset.

        Args:
            asset_idx: Asset index
            position_change: Change in position
            price: Current price
        """
        # Apply slippage
        if position_change > 0:
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)

        # Calculate trade value
        trade_value = abs(position_change) * self.initial_capital * self.leverage

        # Calculate fees
        fee = trade_value * self.taker_fee
        self.total_fees += fee
        self.cash -= fee

        # Update position
        old_position = self.positions[asset_idx]
        self.positions[asset_idx] += position_change
        self.positions[asset_idx] = np.clip(
            self.positions[asset_idx],
            -self.max_position_per_asset,
            self.max_position_per_asset,
        )

        # Update entry price
        if abs(self.positions[asset_idx]) > 0.01:
            if position_change * self.positions[asset_idx] > 0:
                if abs(old_position) > 0.01:
                    total_value = abs(old_position) * self.entry_prices[asset_idx] + abs(position_change) * execution_price
                    self.entry_prices[asset_idx] = total_value / abs(self.positions[asset_idx])
                else:
                    self.entry_prices[asset_idx] = execution_price
            else:
                self.entry_prices[asset_idx] = execution_price
        else:
            self.entry_prices[asset_idx] = 0.0

        self.trade_counts[asset_idx] += 1

    def _update_portfolio_value(self, current_prices: np.ndarray) -> None:
        """Update portfolio value based on current positions and prices.

        Args:
            current_prices: Current prices for all assets
        """
        total_pnl = 0.0

        for i in range(self.n_assets):
            if abs(self.positions[i]) > 0.01 and self.entry_prices[i] > 0:
                price_change_pct = (current_prices[i] - self.entry_prices[i]) / self.entry_prices[i]
                position_value = abs(self.positions[i]) * self.initial_capital * self.leverage
                pnl = np.sign(self.positions[i]) * price_change_pct * position_value
                total_pnl += pnl

        self.portfolio_value = self.cash + self.initial_capital + total_pnl

    def _calculate_reward(self, actions: np.ndarray) -> float:
        """Calculate reward for the current step.

        Args:
            actions: Actions taken

        Returns:
            Reward value
        """
        # Portfolio return
        returns = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        reward = returns * self.reward_scaling

        # Transaction penalty
        turnover = np.sum(np.abs(actions - self.prev_positions))
        reward -= self.transaction_penalty * turnover

        # Drawdown penalty
        if self.peak_value > 0:
            current_dd = (self.peak_value - self.portfolio_value) / self.peak_value
            reward -= self.drawdown_penalty * current_dd

        # Position change penalty (penalize action changes, not just turnover)
        action_change = np.sum(np.abs(actions - self.prev_positions))
        if action_change > 0.1:  # Only penalize significant changes
            reward -= self.transaction_penalty * action_change * 5  # 5x multiplier
        
        # Holding bonus: reward for staying in a position
        total_position = np.sum(np.abs(self.positions))
        if total_position > 0.1:
            # Small bonus for holding positions
            holding_bonus = 0.001 * total_position
            reward += holding_bonus

        # Sharpe bonus
        if len(self.returns_history) > 20:
            mean_return = np.mean(self.returns_history[-20:])
            std_return = np.std(self.returns_history[-20:]) + 1e-8
            sharpe = mean_return / std_return
            reward += 0.01 * sharpe

        # Diversification bonus (reward for spreading positions)
        position_concentration = np.std(np.abs(self.positions))
        reward += 0.001 * (1 - position_concentration)

        return float(reward)

    def _get_observation(self) -> dict:
        """Get current observation.

        Returns:
            Observation dict
        """
        # Clamp to valid range
        step = min(self.current_step, self.n_steps - 1)
        
        # Market features for each asset
        market_obs = np.zeros(
            (self.n_assets, self.lookback, self.n_features), dtype=np.float32
        )
        for i, symbol in enumerate(self.symbols):
            market_obs[i] = self.features_dict[symbol][step]

        # Portfolio state
        cash_pct = self.cash / self.initial_capital
        value_pct = self.portfolio_value / self.initial_capital

        portfolio_obs = np.concatenate([
            self.positions.astype(np.float32),
            np.array([cash_pct, value_pct], dtype=np.float32),
        ])

        return {
            "market": market_obs,
            "portfolio": portfolio_obs,
        }

    def _get_info(self) -> dict:
        """Get current info dict.

        Returns:
            Info dict
        """
        current_prices = self._get_current_prices()

        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "positions": self.positions.copy(),
            "cash": self.cash,
            "total_return": (self.portfolio_value - self.initial_capital) / self.initial_capital,
            "max_drawdown": (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0,
            "trade_counts": self.trade_counts.copy(),
            "total_fees": self.total_fees,
            "prices": current_prices,
        }

    def render(self) -> Optional[str]:
        """Render the current state."""
        info = self._get_info()

        position_str = " | ".join(
            f"{self.symbols[i]}: {self.positions[i]:.2f}"
            for i in range(min(3, self.n_assets))
        )

        output = (
            f"Step: {info['step']} | "
            f"Value: ${info['portfolio_value']:.2f} | "
            f"Return: {info['total_return']*100:.2f}% | "
            f"Positions: [{position_str}...]"
        )

        if self.render_mode == "human":
            print(output)
        elif self.render_mode == "ansi":
            return output

        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    def get_performance_metrics(self) -> dict:
        """Calculate performance metrics for the episode.

        Returns:
            Dict of performance metrics
        """
        if len(self.returns_history) == 0:
            return {}

        returns = np.array(self.returns_history)
        values = np.array(self.value_history)

        minutes_per_year = 365.25 * 24 * 60

        # Sharpe Ratio
        sharpe = np.sqrt(minutes_per_year) * np.mean(returns) / (np.std(returns) + 1e-8)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            sortino = np.sqrt(minutes_per_year) * np.mean(returns) / (np.std(downside_returns) + 1e-8)
        else:
            sortino = np.inf

        # Max Drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / (peak + 1e-8)
        max_dd = np.max(drawdown)

        # Calmar Ratio
        total_return = (values[-1] - values[0]) / values[0]
        calmar = total_return / (max_dd + 1e-8)

        # Win Rate
        win_rate = np.mean(returns > 0)

        # Per-asset metrics
        position_history = np.array(self.position_history)
        avg_positions = np.mean(np.abs(position_history), axis=0)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "trade_counts": self.trade_counts.tolist(),
            "total_fees": self.total_fees,
            "final_value": values[-1] if len(values) > 0 else self.initial_capital,
            "avg_positions_per_asset": {
                self.symbols[i]: avg_positions[i]
                for i in range(self.n_assets)
            },
        }
