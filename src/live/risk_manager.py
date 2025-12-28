"""Risk management module for live trading."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from loguru import logger


@dataclass
class RiskConfig:
    """Risk management configuration."""
    
    # Position limits
    max_position_per_asset: float = 0.2  # Max 20% per asset
    max_total_exposure: float = 0.8  # Max 80% total exposure
    max_leverage: int = 3
    
    # Loss limits
    max_drawdown: float = 0.2  # 20% max drawdown
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    position_stop_loss: float = 0.02  # 2% per position
    
    # Trading limits
    max_trades_per_hour: int = 20
    min_trade_interval_seconds: int = 10
    max_order_value: float = 50.0  # Max USD per order
    
    # Circuit breakers
    consecutive_loss_limit: int = 5
    volatility_pause_threshold: float = 0.05  # 5% volatility triggers pause
    pause_duration_minutes: int = 30


@dataclass
class PositionInfo:
    """Information about a single position."""
    
    symbol: str
    size: float  # Positive for long, negative for short
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def pnl_pct(self) -> float:
        """Calculate PnL as percentage of entry."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * np.sign(self.size)


class RiskManager:
    """Risk management for live trading."""
    
    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        initial_capital: float = 100.0,
    ):
        """Initialize risk manager.
        
        Args:
            config: Risk configuration
            initial_capital: Starting capital
        """
        self.config = config or RiskConfig()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # State tracking
        self.positions: dict[str, PositionInfo] = {}
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.daily_start_time = datetime.utcnow()
        
        # Trade tracking
        self.trades_this_hour: list[datetime] = []
        self.last_trade_time: Optional[datetime] = None
        self.consecutive_losses = 0
        
        # Circuit breaker state
        self.is_paused = False
        self.pause_until: Optional[datetime] = None
        self.pause_reason: str = ""
        
        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        logger.info("Risk manager initialized")

    def update_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        current_price: float,
    ) -> None:
        """Update or create a position.
        
        Args:
            symbol: Trading symbol
            size: Position size (positive=long, negative=short)
            entry_price: Entry price
            current_price: Current market price
        """
        if abs(size) < 1e-8:
            # Position closed
            if symbol in self.positions:
                del self.positions[symbol]
            return
        
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.size = size
            pos.current_price = current_price
            pos.unrealized_pnl = (current_price - pos.entry_price) * size
        else:
            self.positions[symbol] = PositionInfo(
                symbol=symbol,
                size=size,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=(current_price - entry_price) * size,
            )

    def update_capital(self, new_capital: float) -> None:
        """Update current capital.
        
        Args:
            new_capital: New capital value
        """
        old_capital = self.current_capital
        self.current_capital = new_capital
        
        # Update peak
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        
        # Reset daily tracking at midnight
        now = datetime.utcnow()
        if now.date() > self.daily_start_time.date():
            self.daily_start_capital = old_capital
            self.daily_start_time = now

    def check_trade_allowed(
        self,
        symbol: str,
        target_position: float,
        current_price: float,
    ) -> tuple[bool, str]:
        """Check if a trade is allowed by risk rules.
        
        Args:
            symbol: Trading symbol
            target_position: Target position size (-1 to 1)
            current_price: Current market price
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check pause
        if self.is_paused:
            if self.pause_until and datetime.utcnow() >= self.pause_until:
                self.is_paused = False
                self.pause_reason = ""
                logger.info("Trading resumed after pause")
            else:
                return False, f"Trading paused: {self.pause_reason}"
        
        # Check drawdown
        if self.peak_capital > 0:
            current_dd = (self.peak_capital - self.current_capital) / self.peak_capital
            if current_dd >= self.config.max_drawdown:
                self._trigger_pause(f"Max drawdown exceeded: {current_dd*100:.1f}%")
                return False, "Max drawdown exceeded"
        
        # Check daily loss
        if self.daily_start_capital > 0:
            daily_loss = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
            if daily_loss >= self.config.daily_loss_limit:
                self._trigger_pause(f"Daily loss limit exceeded: {daily_loss*100:.1f}%")
                return False, "Daily loss limit exceeded"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.config.consecutive_loss_limit:
            self._trigger_pause(f"Consecutive losses: {self.consecutive_losses}")
            return False, "Too many consecutive losses"
        
        # Check trade frequency
        now = datetime.utcnow()
        
        # Clean old trades
        hour_ago = now - timedelta(hours=1)
        self.trades_this_hour = [t for t in self.trades_this_hour if t > hour_ago]
        
        if len(self.trades_this_hour) >= self.config.max_trades_per_hour:
            return False, "Max trades per hour exceeded"
        
        # Check minimum interval
        if self.last_trade_time:
            seconds_since_last = (now - self.last_trade_time).total_seconds()
            if seconds_since_last < self.config.min_trade_interval_seconds:
                return False, f"Min trade interval not met ({seconds_since_last:.1f}s)"
        
        # Check position size
        if abs(target_position) > self.config.max_position_per_asset:
            return False, f"Position size {abs(target_position)} exceeds max {self.config.max_position_per_asset}"
        
        # Check total exposure
        total_exposure = sum(abs(p.size) for p in self.positions.values())
        current_pos_size = abs(self.positions.get(symbol, PositionInfo(symbol, 0, 0)).size)
        new_exposure = total_exposure - current_pos_size + abs(target_position)
        
        if new_exposure > self.config.max_total_exposure:
            return False, f"Total exposure {new_exposure} exceeds max {self.config.max_total_exposure}"
        
        # Check order value
        order_value = abs(target_position) * self.current_capital
        if order_value > self.config.max_order_value:
            return False, f"Order value ${order_value:.2f} exceeds max ${self.config.max_order_value}"
        
        return True, "OK"

    def check_stop_loss(self, symbol: str) -> tuple[bool, float]:
        """Check if position should be stopped out.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (should_stop, stop_price)
        """
        if symbol not in self.positions:
            return False, 0.0
        
        pos = self.positions[symbol]
        pnl_pct = pos.pnl_pct
        
        if pnl_pct <= -self.config.position_stop_loss:
            logger.warning(f"STOP LOSS triggered for {symbol}: {pnl_pct*100:.2f}%")
            return True, pos.current_price
        
        return False, 0.0

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade.
        
        Args:
            pnl: Profit/loss from the trade
        """
        now = datetime.utcnow()
        self.trades_this_hour.append(now)
        self.last_trade_time = now
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def _trigger_pause(self, reason: str) -> None:
        """Trigger trading pause.
        
        Args:
            reason: Reason for pause
        """
        self.is_paused = True
        self.pause_until = datetime.utcnow() + timedelta(minutes=self.config.pause_duration_minutes)
        self.pause_reason = reason
        logger.warning(f"TRADING PAUSED: {reason} (until {self.pause_until})")

    def adjust_position_for_risk(
        self,
        symbol: str,
        target_position: float,
    ) -> float:
        """Adjust target position based on risk constraints.
        
        Args:
            symbol: Trading symbol
            target_position: Original target position
            
        Returns:
            Adjusted position (may be smaller or zero)
        """
        # Clamp to max position
        adjusted = np.clip(
            target_position,
            -self.config.max_position_per_asset,
            self.config.max_position_per_asset,
        )
        
        # Reduce position if approaching max drawdown
        if self.peak_capital > 0:
            dd = (self.peak_capital - self.current_capital) / self.peak_capital
            if dd > self.config.max_drawdown * 0.7:  # Start reducing at 70% of max
                reduction_factor = 1 - (dd / self.config.max_drawdown)
                adjusted *= reduction_factor
                logger.info(f"Position reduced by {(1-reduction_factor)*100:.1f}% due to drawdown")
        
        return adjusted

    def get_metrics(self) -> dict:
        """Get risk management metrics.
        
        Returns:
            Dict of metrics
        """
        return {
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "current_drawdown": (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0,
            "daily_pnl": (self.current_capital - self.daily_start_capital) / self.daily_start_capital if self.daily_start_capital > 0 else 0,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            "total_pnl": self.total_pnl,
            "consecutive_losses": self.consecutive_losses,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "active_positions": len(self.positions),
            "total_exposure": sum(abs(p.size) for p in self.positions.values()),
        }

    def get_position_summary(self) -> list[dict]:
        """Get summary of all positions.
        
        Returns:
            List of position dicts
        """
        return [
            {
                "symbol": pos.symbol,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "pnl_pct": pos.pnl_pct * 100,
                "age_minutes": (datetime.utcnow() - pos.entry_time).total_seconds() / 60,
            }
            for pos in self.positions.values()
        ]
