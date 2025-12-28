"""Market regime detection and state classification."""

from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger


class MarketRegime(Enum):
    """Market regime classification."""
    
    STRONG_BULL = "strong_bull"      # Strong uptrend, low volatility
    BULL = "bull"                     # Uptrend
    RANGE = "range"                   # Sideways, no clear trend
    BEAR = "bear"                     # Downtrend
    STRONG_BEAR = "strong_bear"       # Strong downtrend, high volatility
    HIGH_VOLATILITY = "high_vol"      # Extreme volatility, no direction


class RegimeDetector:
    """Detect market regime based on price and volatility patterns.
    
    Uses multiple indicators:
    - Trend: SMA crossovers, ADX
    - Volatility: ATR, rolling std, Bollinger width
    - Momentum: RSI, MACD histogram
    """
    
    def __init__(
        self,
        trend_short_window: int = 20,
        trend_long_window: int = 50,
        volatility_window: int = 20,
        atr_window: int = 14,
        volatility_threshold_high: float = 2.0,
        volatility_threshold_low: float = 0.5,
        trend_strength_threshold: float = 25,
    ):
        """Initialize regime detector.
        
        Args:
            trend_short_window: Short-term trend window
            trend_long_window: Long-term trend window
            volatility_window: Volatility calculation window
            atr_window: ATR calculation window
            volatility_threshold_high: High volatility threshold (multiplier of mean)
            volatility_threshold_low: Low volatility threshold (multiplier of mean)
            trend_strength_threshold: ADX threshold for trend strength
        """
        self.trend_short_window = trend_short_window
        self.trend_long_window = trend_long_window
        self.volatility_window = volatility_window
        self.atr_window = atr_window
        self.volatility_threshold_high = volatility_threshold_high
        self.volatility_threshold_low = volatility_threshold_low
        self.trend_strength_threshold = trend_strength_threshold
        
        # State
        self.current_regime: Optional[MarketRegime] = None
        self.regime_history: list[dict] = []
        self.volatility_baseline: Optional[float] = None

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Detected market regime
        """
        if len(df) < self.trend_long_window + 20:
            return MarketRegime.RANGE
        
        # Calculate indicators
        trend_score = self._calculate_trend_score(df)
        volatility_level = self._calculate_volatility_level(df)
        momentum = self._calculate_momentum(df)
        
        # Classify regime
        regime = self._classify_regime(trend_score, volatility_level, momentum)
        
        # Update state
        self.current_regime = regime
        self.regime_history.append({
            "timestamp": df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else pd.Timestamp.now(),
            "regime": regime,
            "trend_score": trend_score,
            "volatility_level": volatility_level,
            "momentum": momentum,
        })
        
        # Keep only last 1000 entries
        self.regime_history = self.regime_history[-1000:]
        
        return regime

    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """Calculate trend score from -1 (bearish) to +1 (bullish).
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Trend score
        """
        close = df["close"].values
        
        # SMA crossover
        sma_short = np.mean(close[-self.trend_short_window:])
        sma_long = np.mean(close[-self.trend_long_window:])
        sma_score = (sma_short - sma_long) / sma_long
        
        # Price vs SMA
        current_price = close[-1]
        price_vs_sma = (current_price - sma_short) / sma_short
        
        # Simple trend (recent price movement)
        returns_20 = (close[-1] - close[-20]) / close[-20]
        
        # ADX-like trend strength (simplified)
        high = df["high"].values
        low = df["low"].values
        
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        atr = np.mean(tr[-self.atr_window:])
        plus_di = np.mean(plus_dm[-self.atr_window:]) / (atr + 1e-8)
        minus_di = np.mean(minus_dm[-self.atr_window:]) / (atr + 1e-8)
        
        di_diff = plus_di - minus_di
        di_sum = plus_di + minus_di + 1e-8
        adx_direction = di_diff / di_sum
        
        # Combine scores
        trend_score = 0.3 * sma_score * 10 + 0.3 * price_vs_sma * 10 + 0.2 * returns_20 * 10 + 0.2 * adx_direction
        
        return np.clip(trend_score, -1, 1)

    def _calculate_volatility_level(self, df: pd.DataFrame) -> float:
        """Calculate volatility level (normalized).
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Volatility level (1.0 = normal, >2 = high, <0.5 = low)
        """
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        # Returns volatility
        returns = np.diff(close) / close[:-1]
        current_vol = np.std(returns[-self.volatility_window:]) * np.sqrt(1440)  # Annualized
        
        # ATR-based volatility
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        current_atr = np.mean(tr[-self.atr_window:])
        atr_pct = current_atr / close[-1]
        
        # Update baseline
        if self.volatility_baseline is None:
            self.volatility_baseline = np.std(returns[-self.volatility_window * 5:]) * np.sqrt(1440)
        else:
            # Exponential moving average
            self.volatility_baseline = 0.99 * self.volatility_baseline + 0.01 * current_vol
        
        # Normalized volatility
        vol_level = current_vol / (self.volatility_baseline + 1e-8)
        
        return vol_level

    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum indicator.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Momentum score from -1 to 1
        """
        close = df["close"].values
        
        # RSI
        deltas = np.diff(close)
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        rsi_normalized = (rsi - 50) / 50
        
        # Rate of change
        roc_5 = (close[-1] - close[-5]) / close[-5]
        roc_10 = (close[-1] - close[-10]) / close[-10]
        
        momentum = 0.5 * rsi_normalized + 0.25 * roc_5 * 10 + 0.25 * roc_10 * 5
        
        return np.clip(momentum, -1, 1)

    def _classify_regime(
        self,
        trend_score: float,
        volatility_level: float,
        momentum: float,
    ) -> MarketRegime:
        """Classify market regime based on indicators.
        
        Args:
            trend_score: Trend score (-1 to 1)
            volatility_level: Volatility level
            momentum: Momentum score (-1 to 1)
            
        Returns:
            Market regime
        """
        # High volatility regime
        if volatility_level > self.volatility_threshold_high:
            if trend_score > 0.3 and momentum > 0.3:
                return MarketRegime.STRONG_BULL
            elif trend_score < -0.3 and momentum < -0.3:
                return MarketRegime.STRONG_BEAR
            else:
                return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility (ranging)
        if volatility_level < self.volatility_threshold_low:
            if abs(trend_score) < 0.2 and abs(momentum) < 0.2:
                return MarketRegime.RANGE
        
        # Trend-based classification
        if trend_score > 0.5 and momentum > 0.3:
            return MarketRegime.STRONG_BULL
        elif trend_score > 0.2:
            return MarketRegime.BULL
        elif trend_score < -0.5 and momentum < -0.3:
            return MarketRegime.STRONG_BEAR
        elif trend_score < -0.2:
            return MarketRegime.BEAR
        else:
            return MarketRegime.RANGE

    def get_position_multiplier(self, regime: Optional[MarketRegime] = None) -> float:
        """Get position size multiplier based on regime.
        
        Args:
            regime: Market regime (uses current if not specified)
            
        Returns:
            Position multiplier (0.0 to 1.5)
        """
        regime = regime or self.current_regime
        
        multipliers = {
            MarketRegime.STRONG_BULL: 1.5,
            MarketRegime.BULL: 1.2,
            MarketRegime.RANGE: 0.5,
            MarketRegime.BEAR: 0.8,
            MarketRegime.STRONG_BEAR: 0.3,
            MarketRegime.HIGH_VOLATILITY: 0.2,
        }
        
        return multipliers.get(regime, 1.0)

    def get_recommended_strategy(self, regime: Optional[MarketRegime] = None) -> str:
        """Get recommended trading strategy for regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Strategy recommendation
        """
        regime = regime or self.current_regime
        
        strategies = {
            MarketRegime.STRONG_BULL: "Trend following, long bias, trailing stops",
            MarketRegime.BULL: "Trend following with pullback entries",
            MarketRegime.RANGE: "Mean reversion, tight stops, reduced size",
            MarketRegime.BEAR: "Short bias, quick profits, tight stops",
            MarketRegime.STRONG_BEAR: "Short only or stay flat, hedge positions",
            MarketRegime.HIGH_VOLATILITY: "Reduced size, wider stops, quick exits",
        }
        
        return strategies.get(regime, "Unknown regime")

    def get_regime_stats(self) -> dict:
        """Get statistics on regime history.
        
        Returns:
            Dict with regime statistics
        """
        if not self.regime_history:
            return {}
        
        regimes = [r["regime"] for r in self.regime_history]
        
        # Count occurrences
        counts = {}
        for regime in MarketRegime:
            counts[regime.value] = sum(1 for r in regimes if r == regime)
        
        total = len(regimes)
        
        return {
            "current_regime": self.current_regime.value if self.current_regime else None,
            "regime_distribution": {k: v / total for k, v in counts.items()},
            "total_observations": total,
            "last_10_regimes": [r.value for r in regimes[-10:]],
        }

    def analyze_multi_timeframe(
        self,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> dict:
        """Analyze regime across multiple timeframes.
        
        Args:
            df_1m: 1-minute OHLCV data
            df_5m: 5-minute OHLCV data  
            df_15m: 15-minute OHLCV data
            
        Returns:
            Multi-timeframe analysis
        """
        regime_1m = self.detect(df_1m)
        regime_5m = self.detect(df_5m)
        regime_15m = self.detect(df_15m)
        
        # Agreement score
        regimes = [regime_1m, regime_5m, regime_15m]
        is_bullish = [r in (MarketRegime.BULL, MarketRegime.STRONG_BULL) for r in regimes]
        is_bearish = [r in (MarketRegime.BEAR, MarketRegime.STRONG_BEAR) for r in regimes]
        
        if all(is_bullish):
            consensus = "strong_bullish"
            confidence = 1.0
        elif all(is_bearish):
            consensus = "strong_bearish"
            confidence = 1.0
        elif sum(is_bullish) > sum(is_bearish):
            consensus = "bullish"
            confidence = sum(is_bullish) / 3
        elif sum(is_bearish) > sum(is_bullish):
            consensus = "bearish"
            confidence = sum(is_bearish) / 3
        else:
            consensus = "neutral"
            confidence = 0.0
        
        return {
            "regime_1m": regime_1m.value,
            "regime_5m": regime_5m.value,
            "regime_15m": regime_15m.value,
            "consensus": consensus,
            "confidence": confidence,
            "recommended_multiplier": self.get_position_multiplier(regime_15m),
        }
