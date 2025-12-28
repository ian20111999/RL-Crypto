"""Dynamic leverage based on market conditions."""

from typing import Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger

from src.analysis.regime_detector import MarketRegime


@dataclass
class LeverageConfig:
    """Dynamic leverage configuration."""
    
    base_leverage: int = 2
    min_leverage: int = 1
    max_leverage: int = 5
    
    # Regime-based multipliers
    regime_multipliers: dict = None
    
    # Volatility-based adjustment
    volatility_threshold_low: float = 0.01
    volatility_threshold_high: float = 0.03
    
    # Drawdown-based adjustment
    drawdown_threshold: float = 0.1
    
    def __post_init__(self):
        if self.regime_multipliers is None:
            self.regime_multipliers = {
                MarketRegime.STRONG_BULL: 1.5,
                MarketRegime.BULL: 1.2,
                MarketRegime.RANGE: 0.5,
                MarketRegime.BEAR: 0.7,
                MarketRegime.STRONG_BEAR: 0.3,
                MarketRegime.HIGH_VOLATILITY: 0.2,
            }


class DynamicLeverage:
    """Dynamically adjust leverage based on market conditions.
    
    Factors:
    - Market regime (bull/bear/range)
    - Current volatility
    - Portfolio drawdown
    - Consecutive losses
    """
    
    def __init__(self, config: Optional[LeverageConfig] = None):
        """Initialize dynamic leverage.
        
        Args:
            config: Leverage configuration
        """
        self.config = config or LeverageConfig()
        
        # State
        self.current_leverage = self.config.base_leverage
        self.current_regime: Optional[MarketRegime] = None
        self.current_volatility: float = 0.0
        self.current_drawdown: float = 0.0
        self.consecutive_losses: int = 0
        
        # History
        self.leverage_history: list[dict] = []
        
        logger.info(f"DynamicLeverage initialized (base={self.config.base_leverage}x)")

    def update(
        self,
        regime: Optional[MarketRegime] = None,
        volatility: Optional[float] = None,
        drawdown: Optional[float] = None,
        consecutive_losses: Optional[int] = None,
    ) -> int:
        """Update and calculate new leverage.
        
        Args:
            regime: Current market regime
            volatility: Current volatility (daily)
            drawdown: Current portfolio drawdown
            consecutive_losses: Number of consecutive losing trades
            
        Returns:
            New leverage value
        """
        # Update state
        if regime is not None:
            self.current_regime = regime
        if volatility is not None:
            self.current_volatility = volatility
        if drawdown is not None:
            self.current_drawdown = drawdown
        if consecutive_losses is not None:
            self.consecutive_losses = consecutive_losses
        
        # Calculate new leverage
        new_leverage = self._calculate_leverage()
        
        # Log if changed
        if new_leverage != self.current_leverage:
            logger.info(
                f"Leverage changed: {self.current_leverage}x -> {new_leverage}x "
                f"(regime={self.current_regime}, vol={self.current_volatility:.4f}, "
                f"dd={self.current_drawdown:.2%})"
            )
        
        self.current_leverage = new_leverage
        
        # Record history
        self.leverage_history.append({
            "leverage": new_leverage,
            "regime": self.current_regime,
            "volatility": self.current_volatility,
            "drawdown": self.current_drawdown,
        })
        self.leverage_history = self.leverage_history[-1000:]
        
        return new_leverage

    def _calculate_leverage(self) -> int:
        """Calculate leverage based on current conditions.
        
        Returns:
            Leverage value
        """
        multiplier = 1.0
        
        # 1. Regime-based adjustment
        if self.current_regime:
            regime_mult = self.config.regime_multipliers.get(self.current_regime, 1.0)
            multiplier *= regime_mult
        
        # 2. Volatility-based adjustment
        if self.current_volatility > 0:
            if self.current_volatility > self.config.volatility_threshold_high:
                # High volatility: reduce leverage
                vol_mult = self.config.volatility_threshold_high / self.current_volatility
                multiplier *= np.clip(vol_mult, 0.2, 1.0)
            elif self.current_volatility < self.config.volatility_threshold_low:
                # Low volatility: can increase slightly
                multiplier *= 1.2
        
        # 3. Drawdown-based adjustment
        if self.current_drawdown > self.config.drawdown_threshold:
            dd_mult = 1 - (self.current_drawdown - self.config.drawdown_threshold)
            multiplier *= np.clip(dd_mult, 0.3, 1.0)
        
        # 4. Consecutive losses adjustment
        if self.consecutive_losses >= 3:
            loss_mult = 1 - (self.consecutive_losses - 2) * 0.1
            multiplier *= np.clip(loss_mult, 0.5, 1.0)
        
        # Calculate final leverage
        leverage = self.config.base_leverage * multiplier
        leverage = np.clip(leverage, self.config.min_leverage, self.config.max_leverage)
        
        return int(round(leverage))

    def get_leverage(self) -> int:
        """Get current leverage.
        
        Returns:
            Current leverage
        """
        return self.current_leverage

    def get_stats(self) -> dict:
        """Get leverage statistics.
        
        Returns:
            Stats dict
        """
        if not self.leverage_history:
            return {}
        
        leverages = [h["leverage"] for h in self.leverage_history]
        
        return {
            "current_leverage": self.current_leverage,
            "avg_leverage": np.mean(leverages),
            "min_leverage": min(leverages),
            "max_leverage": max(leverages),
            "changes": sum(1 for i in range(1, len(leverages)) if leverages[i] != leverages[i-1]),
        }


class VolatilityTracker:
    """Track volatility for leverage adjustment."""
    
    def __init__(self, window: int = 60):
        """Initialize volatility tracker.
        
        Args:
            window: Rolling window size
        """
        self.window = window
        self.returns: list[float] = []

    def update(self, price: float) -> Optional[float]:
        """Update with new price and calculate volatility.
        
        Args:
            price: Current price
            
        Returns:
            Current volatility or None
        """
        if hasattr(self, "_last_price"):
            ret = (price - self._last_price) / self._last_price
            self.returns.append(ret)
            self.returns = self.returns[-self.window:]
        
        self._last_price = price
        
        if len(self.returns) >= 20:
            return np.std(self.returns) * np.sqrt(1440)  # Daily vol
        
        return None
