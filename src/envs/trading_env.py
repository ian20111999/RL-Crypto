"""Single-asset trading environment for RL training."""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger


class TradingEnv(gym.Env):
    """Gymnasium environment for single-asset cryptocurrency futures trading.

    This environment simulates trading a single cryptocurrency futures contract
    with continuous position sizing from -1 (full short) to +1 (full long).
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        initial_capital: float = 100.0,
        leverage: int = 2,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        slippage: float = 0.0001,
        episode_length: Optional[int] = None,
        reward_scaling: float = 100.0,
        transaction_penalty: float = 0.001,
        drawdown_penalty: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        """Initialize trading environment.

        Args:
            features: Feature array of shape (n_steps, lookback, n_features)
            prices: Price array of shape (n_steps, 4) for OHLC
            initial_capital: Starting capital in USDT
            leverage: Leverage multiplier
            maker_fee: Maker fee rate
            taker_fee: Taker fee rate
            slippage: Slippage rate
            episode_length: Maximum steps per episode
            reward_scaling: Reward multiplier
            transaction_penalty: Penalty for position changes
            drawdown_penalty: Penalty coefficient for drawdown
            render_mode: Rendering mode
        """
        super().__init__()

        self.features = features
        self.prices = prices  # OHLC
        self.n_steps = len(features)
        self.lookback = features.shape[1]
        self.n_features = features.shape[2]

        # Trading parameters
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.episode_length = episode_length or self.n_steps

        # Reward parameters
        self.reward_scaling = reward_scaling
        self.transaction_penalty = transaction_penalty
        self.drawdown_penalty = drawdown_penalty

        self.render_mode = render_mode

        # Observation space: market features + portfolio state
        # Portfolio state: [position, unrealized_pnl_pct, cash_pct, portfolio_value_pct]
        self.observation_space = spaces.Dict({
            "market": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.lookback, self.n_features),
                dtype=np.float32,
            ),
            "portfolio": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float32,
            ),
        })

        # Action space: continuous position from -1 (short) to +1 (long)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Initialize state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.current_step = 0
        self.cash = self.initial_capital
        self.position = 0.0  # Current position (-1 to 1)
        self.entry_price = 0.0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        self.prev_position = 0.0

        # Tracking
        self.returns_history: list[float] = []
        self.position_history: list[float] = []
        self.value_history: list[float] = []
        self.action_history: list[float] = []
        self.trade_count = 0
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
            options: Additional options (can include 'start_step')

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        self._reset_state()

        # Optional: start from a random point in the data
        if options and "start_step" in options:
            self.current_step = options["start_step"]
        elif self.n_steps > self.episode_length:
            max_start = self.n_steps - self.episode_length
            self.current_step = self.np_random.integers(0, max_start)

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: Target position array of shape (1,)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Clip action to valid range
        target_position = np.clip(action[0], -1.0, 1.0)

        # Get current price
        current_price = self.prices[self.current_step][3]  # Close price

        # Execute trade if position changed
        position_change = target_position - self.position
        if abs(position_change) > 0.01:  # Threshold for trade execution
            self._execute_trade(position_change, current_price)

        # Update portfolio value
        self._update_portfolio_value(current_price)

        # Calculate reward
        reward = self._calculate_reward(target_position)

        # Update tracking
        self.returns_history.append(
            (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        )
        self.position_history.append(self.position)
        self.value_history.append(self.portfolio_value)
        self.action_history.append(target_position)

        # Update previous values
        self.prev_portfolio_value = self.portfolio_value
        self.prev_position = self.position
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Advance step
        self.current_step += 1

        # Check termination conditions
        terminated = False
        truncated = False

        # Terminate if bankrupt
        if self.portfolio_value <= 0:
            terminated = True

        # Terminate if max drawdown exceeded
        if self.peak_value > 0:
            current_dd = (self.peak_value - self.portfolio_value) / self.peak_value
            if current_dd > 0.5:  # 50% max drawdown
                terminated = True

        # Truncate if end of episode or data
        if self.current_step >= min(self.n_steps, self.episode_length):
            truncated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _execute_trade(self, position_change: float, price: float) -> None:
        """Execute a trade with fees and slippage.

        Args:
            position_change: Change in position (-2 to +2 range)
            price: Current market price
        """
        # Apply slippage
        if position_change > 0:  # Buying
            execution_price = price * (1 + self.slippage)
        else:  # Selling
            execution_price = price * (1 - self.slippage)

        # Calculate trade value
        trade_value = abs(position_change) * self.initial_capital * self.leverage

        # Calculate fees (use taker fee for market orders)
        fee = trade_value * self.taker_fee
        self.total_fees += fee
        self.cash -= fee

        # Update position
        self.position += position_change
        self.position = np.clip(self.position, -1.0, 1.0)

        # Update entry price (weighted average)
        if abs(self.position) > 0.01:
            if position_change * self.position > 0:  # Same direction
                total_value = abs(self.prev_position) * self.entry_price + abs(position_change) * execution_price
                self.entry_price = total_value / abs(self.position)
            else:  # Opposite direction
                self.entry_price = execution_price

        self.trade_count += 1

    def _update_portfolio_value(self, current_price: float) -> None:
        """Update portfolio value based on current position and price.

        Args:
            current_price: Current market price
        """
        if abs(self.position) > 0.01 and self.entry_price > 0:
            # Calculate PnL from leveraged position
            price_change_pct = (current_price - self.entry_price) / self.entry_price
            position_value = abs(self.position) * self.initial_capital * self.leverage

            # PnL is position direction * price change * position value
            pnl = np.sign(self.position) * price_change_pct * position_value

            self.portfolio_value = self.cash + self.initial_capital + pnl
        else:
            self.portfolio_value = self.cash + self.initial_capital

    def _calculate_reward(self, action: float) -> float:
        """Calculate reward for the current step.

        Args:
            action: Action taken

        Returns:
            Reward value
        """
        # 1. Portfolio return
        returns = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        reward = returns * self.reward_scaling

        # 2. Transaction penalty
        turnover = abs(action - self.prev_position)
        reward -= self.transaction_penalty * turnover

        # 3. Drawdown penalty
        if self.peak_value > 0:
            current_dd = (self.peak_value - self.portfolio_value) / self.peak_value
            reward -= self.drawdown_penalty * current_dd

        # 4. Sharpe-like bonus (only after enough history)
        if len(self.returns_history) > 20:
            mean_return = np.mean(self.returns_history[-20:])
            std_return = np.std(self.returns_history[-20:]) + 1e-8
            sharpe = mean_return / std_return
            reward += 0.01 * sharpe

        return float(reward)

    def _get_observation(self) -> dict:
        """Get current observation.

        Returns:
            Observation dict
        """
        # Market features
        market_obs = self.features[self.current_step].astype(np.float32)

        # Portfolio state
        unrealized_pnl_pct = (self.portfolio_value - self.initial_capital) / self.initial_capital
        cash_pct = self.cash / self.initial_capital
        value_pct = self.portfolio_value / self.initial_capital

        portfolio_obs = np.array([
            self.position,
            unrealized_pnl_pct,
            cash_pct,
            value_pct,
        ], dtype=np.float32)

        return {
            "market": market_obs,
            "portfolio": portfolio_obs,
        }

    def _get_info(self) -> dict:
        """Get current info dict.

        Returns:
            Info dict
        """
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "cash": self.cash,
            "total_return": (self.portfolio_value - self.initial_capital) / self.initial_capital,
            "max_drawdown": (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0,
            "trade_count": self.trade_count,
            "total_fees": self.total_fees,
        }

    def render(self) -> Optional[str]:
        """Render the current state.

        Returns:
            String representation if mode is 'ansi'
        """
        info = self._get_info()

        output = (
            f"Step: {info['step']} | "
            f"Value: ${info['portfolio_value']:.2f} | "
            f"Position: {info['position']:.2f} | "
            f"Return: {info['total_return']*100:.2f}% | "
            f"MaxDD: {info['max_drawdown']*100:.2f}%"
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

        # Sharpe Ratio (annualized, assuming 1-minute bars)
        minutes_per_year = 365.25 * 24 * 60
        sharpe = np.sqrt(minutes_per_year) * np.mean(returns) / (np.std(returns) + 1e-8)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = np.sqrt(minutes_per_year) * np.mean(returns) / (np.std(downside_returns) + 1e-8)

        # Max Drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_dd = np.max(drawdown)

        # Calmar Ratio
        total_return = (values[-1] - values[0]) / values[0]
        calmar = total_return / (max_dd + 1e-8)

        # Win Rate
        win_rate = np.mean(returns > 0)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "trade_count": self.trade_count,
            "total_fees": self.total_fees,
            "final_value": values[-1] if len(values) > 0 else self.initial_capital,
        }
