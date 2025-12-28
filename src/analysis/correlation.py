"""Correlation monitor for multi-asset risk management."""

from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger


class CorrelationMonitor:
    """Monitor correlations between assets for risk management.
    
    High correlation between assets means less diversification benefit
    and higher portfolio risk.
    """
    
    def __init__(
        self,
        symbols: list[str],
        window: int = 60,
        high_corr_threshold: float = 0.8,
    ):
        """Initialize correlation monitor.
        
        Args:
            symbols: List of trading symbols
            window: Rolling window for correlation calculation
            high_corr_threshold: Threshold for high correlation warning
        """
        self.symbols = symbols
        self.window = window
        self.high_corr_threshold = high_corr_threshold
        
        # Returns history
        self.returns: dict[str, list[float]] = {s: [] for s in symbols}
        self._last_prices: dict[str, float] = {}
        
        # Correlation matrix cache
        self._corr_matrix: Optional[pd.DataFrame] = None

    def update(self, prices: dict[str, float]) -> Optional[pd.DataFrame]:
        """Update with new prices and recalculate correlation.
        
        Args:
            prices: Dict of symbol to current price
            
        Returns:
            Correlation matrix if enough data, else None
        """
        for symbol, price in prices.items():
            if symbol in self._last_prices:
                ret = (price - self._last_prices[symbol]) / self._last_prices[symbol]
                self.returns[symbol].append(ret)
                self.returns[symbol] = self.returns[symbol][-self.window:]
            self._last_prices[symbol] = price
        
        # Check if enough data
        min_len = min(len(r) for r in self.returns.values())
        if min_len >= 20:
            self._corr_matrix = self._calculate_correlation()
            return self._corr_matrix
        
        return None

    def _calculate_correlation(self) -> pd.DataFrame:
        """Calculate correlation matrix.
        
        Returns:
            Correlation matrix DataFrame
        """
        # Align returns
        min_len = min(len(r) for r in self.returns.values())
        data = {s: self.returns[s][-min_len:] for s in self.symbols}
        
        df = pd.DataFrame(data)
        return df.corr()

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get current correlation matrix.
        
        Returns:
            Correlation matrix or None
        """
        return self._corr_matrix

    def get_high_correlations(self) -> list[tuple[str, str, float]]:
        """Get pairs with high correlation.
        
        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        if self._corr_matrix is None:
            return []
        
        high_corrs = []
        n = len(self.symbols)
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = self._corr_matrix.iloc[i, j]
                if abs(corr) >= self.high_corr_threshold:
                    high_corrs.append((self.symbols[i], self.symbols[j], corr))
        
        return sorted(high_corrs, key=lambda x: abs(x[2]), reverse=True)

    def get_portfolio_exposure_adjustment(
        self,
        target_positions: dict[str, float],
    ) -> dict[str, float]:
        """Adjust positions based on correlation.
        
        Reduces positions for highly correlated assets.
        
        Args:
            target_positions: Target position for each symbol
            
        Returns:
            Adjusted positions
        """
        if self._corr_matrix is None:
            return target_positions
        
        adjusted = target_positions.copy()
        high_corrs = self.get_high_correlations()
        
        for sym1, sym2, corr in high_corrs:
            if sym1 in adjusted and sym2 in adjusted:
                # Reduce both positions by correlation factor
                reduction = 1 - (abs(corr) - self.high_corr_threshold) / (1 - self.high_corr_threshold)
                reduction = max(0.5, reduction)
                
                adjusted[sym1] *= reduction
                adjusted[sym2] *= reduction
                
                logger.debug(
                    f"Reduced {sym1}/{sym2} positions by {(1-reduction)*100:.1f}% "
                    f"(correlation: {corr:.2f})"
                )
        
        return adjusted

    def get_diversification_ratio(self) -> float:
        """Calculate portfolio diversification ratio.
        
        Higher is better (more diversification).
        
        Returns:
            Diversification ratio
        """
        if self._corr_matrix is None:
            return 1.0
        
        # Average off-diagonal correlation
        n = len(self.symbols)
        if n < 2:
            return 1.0
        
        total_corr = 0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_corr += abs(self._corr_matrix.iloc[i, j])
                count += 1
        
        avg_corr = total_corr / count if count > 0 else 0
        
        # Diversification ratio: 1 = perfect diversification, 0 = all correlated
        return 1 - avg_corr

    def get_stats(self) -> dict:
        """Get correlation statistics.
        
        Returns:
            Stats dict
        """
        stats = {
            "diversification_ratio": self.get_diversification_ratio(),
            "high_correlation_pairs": len(self.get_high_correlations()),
        }
        
        if self._corr_matrix is not None:
            # Flatten upper triangle
            n = len(self.symbols)
            all_corrs = []
            for i in range(n):
                for j in range(i + 1, n):
                    all_corrs.append(self._corr_matrix.iloc[i, j])
            
            if all_corrs:
                stats["avg_correlation"] = np.mean(all_corrs)
                stats["max_correlation"] = max(all_corrs)
                stats["min_correlation"] = min(all_corrs)
        
        return stats


class VolatilityRegimeFilter:
    """Filter trades based on volatility regime."""
    
    def __init__(
        self,
        normal_vol: float = 0.02,
        high_vol_threshold: float = 2.0,
        extreme_vol_threshold: float = 3.0,
    ):
        """Initialize volatility filter.
        
        Args:
            normal_vol: Normal daily volatility
            high_vol_threshold: Multiplier for high volatility
            extreme_vol_threshold: Multiplier for extreme volatility
        """
        self.normal_vol = normal_vol
        self.high_vol_threshold = high_vol_threshold
        self.extreme_vol_threshold = extreme_vol_threshold
        
        self.returns: list[float] = []
        self._last_price: Optional[float] = None

    def update(self, price: float) -> float:
        """Update with new price.
        
        Args:
            price: Current price
            
        Returns:
            Current volatility
        """
        if self._last_price is not None:
            ret = (price - self._last_price) / self._last_price
            self.returns.append(ret)
            self.returns = self.returns[-100:]
        
        self._last_price = price
        
        if len(self.returns) >= 20:
            return np.std(self.returns) * np.sqrt(1440)
        return self.normal_vol

    def get_position_multiplier(self, volatility: Optional[float] = None) -> float:
        """Get position size multiplier based on volatility.
        
        Args:
            volatility: Current volatility (or calculated)
            
        Returns:
            Position multiplier (0-1)
        """
        if volatility is None:
            if len(self.returns) >= 20:
                volatility = np.std(self.returns) * np.sqrt(1440)
            else:
                return 1.0
        
        vol_ratio = volatility / self.normal_vol
        
        if vol_ratio >= self.extreme_vol_threshold:
            return 0.2
        elif vol_ratio >= self.high_vol_threshold:
            return 0.5
        elif vol_ratio <= 0.5:
            return 1.2
        else:
            return 1.0

    def should_trade(self, volatility: Optional[float] = None) -> bool:
        """Check if trading is advisable.
        
        Args:
            volatility: Current volatility
            
        Returns:
            True if trading is OK
        """
        multiplier = self.get_position_multiplier(volatility)
        return multiplier > 0
