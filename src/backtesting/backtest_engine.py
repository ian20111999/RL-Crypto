"""Backtesting engine for evaluating trading strategies."""

from pathlib import Path
from typing import Any, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from loguru import logger


class BacktestEngine:
    """Engine for backtesting trading strategies on historical data."""

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100.0,
        leverage: int = 2,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        slippage: float = 0.0001,
    ):
        """Initialize backtest engine.

        Args:
            prices: DataFrame with OHLCV data (must have 'close' column and datetime index)
            initial_capital: Starting capital in USDT
            leverage: Leverage multiplier
            maker_fee: Maker fee rate
            taker_fee: Taker fee rate
            slippage: Slippage rate
        """
        self.prices = prices.copy()
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage

        # Results storage
        self.results: Optional[pd.DataFrame] = None
        self.trades: list[dict] = []
        self.metrics: dict = {}

    def run(
        self,
        strategy: Callable[[pd.DataFrame, int], float],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run backtest with a given strategy.

        Args:
            strategy: Function that takes (prices_df, current_idx) and returns position [-1, 1]
            verbose: Whether to print progress

        Returns:
            DataFrame with backtest results
        """
        n_steps = len(self.prices)

        # Initialize tracking arrays
        positions = np.zeros(n_steps)
        portfolio_values = np.zeros(n_steps)
        cash = np.zeros(n_steps)
        fees_paid = np.zeros(n_steps)

        current_cash = self.initial_capital
        current_position = 0.0
        entry_price = 0.0

        for i in range(n_steps):
            current_price = self.prices["close"].iloc[i]

            # Get strategy signal
            target_position = strategy(self.prices, i)
            target_position = np.clip(target_position, -1.0, 1.0)

            # Execute trade if position changed
            position_change = target_position - current_position
            if abs(position_change) > 0.01:
                # Calculate execution price with slippage
                if position_change > 0:
                    exec_price = current_price * (1 + self.slippage)
                else:
                    exec_price = current_price * (1 - self.slippage)

                # Calculate fees
                trade_value = abs(position_change) * self.initial_capital * self.leverage
                fee = trade_value * self.taker_fee
                current_cash -= fee

                # Record trade
                self.trades.append({
                    "timestamp": self.prices.index[i] if hasattr(self.prices.index[i], 'strftime') else i,
                    "price": exec_price,
                    "position_change": position_change,
                    "fee": fee,
                    "new_position": target_position,
                })

                # Update entry price
                if abs(target_position) > 0.01:
                    if position_change * target_position > 0:
                        if abs(current_position) > 0.01:
                            total = abs(current_position) * entry_price + abs(position_change) * exec_price
                            entry_price = total / abs(target_position)
                        else:
                            entry_price = exec_price
                    else:
                        entry_price = exec_price
                else:
                    entry_price = 0.0

                current_position = target_position

            # Calculate portfolio value
            if abs(current_position) > 0.01 and entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                position_value = abs(current_position) * self.initial_capital * self.leverage
                pnl = np.sign(current_position) * pnl_pct * position_value
                portfolio_value = current_cash + self.initial_capital + pnl
            else:
                portfolio_value = current_cash + self.initial_capital

            # Record state
            positions[i] = current_position
            portfolio_values[i] = portfolio_value
            cash[i] = current_cash
            fees_paid[i] = sum(t["fee"] for t in self.trades)

            if verbose and (i + 1) % (n_steps // 10) == 0:
                logger.info(f"Progress: {(i + 1) / n_steps * 100:.1f}%")

        # Create results DataFrame
        self.results = pd.DataFrame({
            "price": self.prices["close"].values,
            "position": positions,
            "portfolio_value": portfolio_values,
            "cash": cash,
            "fees_paid": fees_paid,
        }, index=self.prices.index)

        # Calculate metrics
        self._calculate_metrics()

        return self.results

    def _calculate_metrics(self) -> None:
        """Calculate performance metrics."""
        if self.results is None:
            return

        values = self.results["portfolio_value"].values
        returns = pd.Series(values).pct_change().dropna()

        # Basic metrics
        total_return = (values[-1] - values[0]) / values[0]
        annual_factor = 365.25 * 24 * 60  # For 1-minute data

        # Sharpe Ratio
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(annual_factor) * returns.mean() / returns.std()
        else:
            sharpe = 0.0

        # Sortino Ratio
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = np.sqrt(annual_factor) * returns.mean() / downside.std()
        else:
            sortino = 0.0

        # Max Drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_dd = np.max(drawdown)

        # Calmar Ratio
        calmar = total_return / max_dd if max_dd > 0 else 0.0

        # Win Rate
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0.0

        # Trade statistics
        n_trades = len(self.trades)
        total_fees = sum(t["fee"] for t in self.trades)

        self.metrics = {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd * 100,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "n_trades": n_trades,
            "total_fees": total_fees,
            "final_value": values[-1],
            "initial_value": values[0],
        }

    def get_metrics(self) -> dict:
        """Get performance metrics.

        Returns:
            Dict of metrics
        """
        return self.metrics.copy()

    def print_metrics(self) -> None:
        """Print performance metrics."""
        if not self.metrics:
            logger.warning("No metrics available. Run backtest first.")
            return

        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Initial Capital:   ${self.metrics['initial_value']:.2f}")
        print(f"Final Value:       ${self.metrics['final_value']:.2f}")
        print(f"Total Return:      {self.metrics['total_return_pct']:.2f}%")
        print("-" * 50)
        print(f"Sharpe Ratio:      {self.metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:     {self.metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown:      {self.metrics['max_drawdown_pct']:.2f}%")
        print(f"Calmar Ratio:      {self.metrics['calmar_ratio']:.2f}")
        print(f"Win Rate:          {self.metrics['win_rate']*100:.2f}%")
        print("-" * 50)
        print(f"Number of Trades:  {self.metrics['n_trades']}")
        print(f"Total Fees:        ${self.metrics['total_fees']:.4f}")
        print("=" * 50 + "\n")

    def plot_results(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot backtest results.

        Args:
            save_path: Path to save figure
            show: Whether to display plot
        """
        if self.results is None:
            logger.warning("No results to plot. Run backtest first.")
            return

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # 1. Price and positions
        ax1 = axes[0]
        ax1.plot(self.results.index, self.results["price"], label="Price", color="blue")
        ax1.set_ylabel("Price", color="blue")
        ax1.legend(loc="upper left")

        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(
            self.results.index,
            self.results["position"],
            alpha=0.3,
            color="green",
            where=self.results["position"] > 0,
            label="Long",
        )
        ax1_twin.fill_between(
            self.results.index,
            self.results["position"],
            alpha=0.3,
            color="red",
            where=self.results["position"] < 0,
            label="Short",
        )
        ax1_twin.set_ylabel("Position")
        ax1_twin.legend(loc="upper right")
        ax1.set_title("Price and Positions")

        # 2. Portfolio Value
        ax2 = axes[1]
        ax2.plot(self.results.index, self.results["portfolio_value"], color="green", label="Portfolio Value")
        ax2.axhline(y=self.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
        ax2.set_ylabel("Value ($)")
        ax2.legend()
        ax2.set_title("Portfolio Value")

        # 3. Drawdown
        ax3 = axes[2]
        values = self.results["portfolio_value"].values
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak * 100
        ax3.fill_between(self.results.index, drawdown, alpha=0.5, color="red")
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_title("Drawdown")

        # 4. Cumulative Returns vs Buy & Hold
        ax4 = axes[3]
        strategy_returns = (self.results["portfolio_value"] / self.initial_capital - 1) * 100
        buyhold_returns = (self.results["price"] / self.results["price"].iloc[0] - 1) * 100

        ax4.plot(self.results.index, strategy_returns, label="Strategy", color="green")
        ax4.plot(self.results.index, buyhold_returns, label="Buy & Hold", color="blue", alpha=0.7)
        ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax4.set_ylabel("Returns (%)")
        ax4.legend()
        ax4.set_title("Cumulative Returns Comparison")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")

        if show:
            plt.show()

        plt.close()

    def compare_strategies(
        self,
        strategies: dict[str, Callable],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Compare multiple strategies.

        Args:
            strategies: Dict mapping strategy name to strategy function
            verbose: Whether to print progress

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for name, strategy in strategies.items():
            logger.info(f"Running backtest for: {name}")

            # Reset state
            self.trades = []
            self.results = None

            # Run backtest
            self.run(strategy, verbose=False)

            metrics = self.metrics.copy()
            metrics["strategy"] = name
            results.append(metrics)

        comparison = pd.DataFrame(results)
        comparison = comparison.set_index("strategy")

        if verbose:
            print("\n" + "=" * 80)
            print("STRATEGY COMPARISON")
            print("=" * 80)
            print(comparison[["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "n_trades"]].to_string())
            print("=" * 80 + "\n")

        return comparison


# Example baseline strategies

def buy_and_hold_strategy(prices: pd.DataFrame, idx: int) -> float:
    """Simple buy and hold strategy.

    Args:
        prices: Price DataFrame
        idx: Current index

    Returns:
        Position (always 1.0 = full long)
    """
    return 1.0


def sma_crossover_strategy(prices: pd.DataFrame, idx: int, fast: int = 10, slow: int = 50) -> float:
    """SMA crossover strategy.

    Args:
        prices: Price DataFrame
        idx: Current index
        fast: Fast SMA period
        slow: Slow SMA period

    Returns:
        Position based on crossover
    """
    if idx < slow:
        return 0.0

    close = prices["close"].iloc[:idx + 1]
    fast_sma = close.rolling(window=fast).mean().iloc[-1]
    slow_sma = close.rolling(window=slow).mean().iloc[-1]

    if fast_sma > slow_sma:
        return 1.0
    elif fast_sma < slow_sma:
        return -1.0
    else:
        return 0.0


def rsi_strategy(prices: pd.DataFrame, idx: int, period: int = 14, oversold: int = 30, overbought: int = 70) -> float:
    """RSI-based mean reversion strategy.

    Args:
        prices: Price DataFrame
        idx: Current index
        period: RSI period
        oversold: Oversold threshold
        overbought: Overbought threshold

    Returns:
        Position based on RSI
    """
    if idx < period + 1:
        return 0.0

    close = prices["close"].iloc[:idx + 1]
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    if rsi < oversold:
        return 1.0  # Buy signal
    elif rsi > overbought:
        return -1.0  # Sell signal
    else:
        return 0.0  # Neutral
