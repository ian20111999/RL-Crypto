"""Alert system for trading notifications."""

import asyncio
import os
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
from loguru import logger


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "ℹ️"
    WARNING = "⚠️"
    ERROR = "🚨"
    CRITICAL = "🔴"
    SUCCESS = "✅"
    TRADE = "📈"


@dataclass
class Alert:
    """Alert message."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def format(self) -> str:
        """Format alert for display."""
        return f"{self.level.value} *{self.title}*\n{self.message}\n_{self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}_"


class TelegramNotifier:
    """Send alerts via Telegram bot."""
    
    API_URL = "https://api.telegram.org"
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token (or from env TELEGRAM_BOT_TOKEN)
            chat_id: Telegram chat ID (or from env TELEGRAM_CHAT_ID)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if self.enabled:
            logger.info("Telegram notifications enabled")
        else:
            logger.warning("Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")

    async def send(self, alert: Alert) -> bool:
        """Send alert via Telegram.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        url = f"{self.API_URL}/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": alert.format(),
            "parse_mode": "Markdown",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        logger.error(f"Telegram error: {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
        
        return False


class SlackNotifier:
    """Send alerts via Slack webhook."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize Slack notifier.
        
        Args:
            webhook_url: Slack webhook URL (or from env SLACK_WEBHOOK_URL)
        """
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.enabled = bool(self.webhook_url)
        
        if self.enabled:
            logger.info("Slack notifications enabled")

    async def send(self, alert: Alert) -> bool:
        """Send alert via Slack.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        payload = {
            "text": alert.format(),
            "username": "RL-Crypto Bot",
            "icon_emoji": alert.level.value,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
        
        return False


class DiscordNotifier:
    """Send alerts via Discord webhook."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize Discord notifier.
        
        Args:
            webhook_url: Discord webhook URL (or from env DISCORD_WEBHOOK_URL)
        """
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.enabled = bool(self.webhook_url)

    async def send(self, alert: Alert) -> bool:
        """Send alert via Discord."""
        if not self.enabled:
            return False
        
        payload = {
            "content": alert.format(),
            "username": "RL-Crypto Bot",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    return resp.status == 204
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
        
        return False


class AlertManager:
    """Manage and route alerts to multiple channels."""
    
    def __init__(
        self,
        enable_telegram: bool = True,
        enable_slack: bool = True,
        enable_discord: bool = True,
        rate_limit_seconds: int = 60,
    ):
        """Initialize alert manager.
        
        Args:
            enable_telegram: Enable Telegram notifications
            enable_slack: Enable Slack notifications
            enable_discord: Enable Discord notifications
            rate_limit_seconds: Minimum seconds between duplicate alerts
        """
        self.notifiers = []
        
        if enable_telegram:
            self.notifiers.append(TelegramNotifier())
        if enable_slack:
            self.notifiers.append(SlackNotifier())
        if enable_discord:
            self.notifiers.append(DiscordNotifier())
        
        self.rate_limit = rate_limit_seconds
        self._last_alerts: dict[str, datetime] = {}
        
        # Alert history
        self.alert_history: list[Alert] = []
        
        # Custom handlers
        self._handlers: list[Callable] = []

    def add_handler(self, handler: Callable) -> None:
        """Add custom alert handler.
        
        Args:
            handler: Callable that takes Alert as argument
        """
        self._handlers.append(handler)

    async def send(self, alert: Alert) -> None:
        """Send alert to all configured channels.
        
        Args:
            alert: Alert to send
        """
        # Rate limiting
        alert_key = f"{alert.level}:{alert.title}"
        now = datetime.utcnow()
        
        if alert_key in self._last_alerts:
            elapsed = (now - self._last_alerts[alert_key]).total_seconds()
            if elapsed < self.rate_limit:
                logger.debug(f"Rate limited alert: {alert_key}")
                return
        
        self._last_alerts[alert_key] = now
        
        # Record
        self.alert_history.append(alert)
        self.alert_history = self.alert_history[-1000:]
        
        # Log
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
            AlertLevel.SUCCESS: logger.success,
            AlertLevel.TRADE: logger.info,
        }.get(alert.level, logger.info)
        
        log_method(f"ALERT: {alert.title} - {alert.message}")
        
        # Send to all notifiers
        tasks = [n.send(alert) for n in self.notifiers if n.enabled]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Call custom handlers
        for handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Handler error: {e}")

    # Convenience methods
    async def info(self, title: str, message: str) -> None:
        await self.send(Alert(AlertLevel.INFO, title, message))
    
    async def warning(self, title: str, message: str) -> None:
        await self.send(Alert(AlertLevel.WARNING, title, message))
    
    async def error(self, title: str, message: str) -> None:
        await self.send(Alert(AlertLevel.ERROR, title, message))
    
    async def critical(self, title: str, message: str) -> None:
        await self.send(Alert(AlertLevel.CRITICAL, title, message))
    
    async def success(self, title: str, message: str) -> None:
        await self.send(Alert(AlertLevel.SUCCESS, title, message))
    
    async def trade(self, title: str, message: str) -> None:
        await self.send(Alert(AlertLevel.TRADE, title, message))

    # Trading-specific alerts
    async def trade_opened(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
    ) -> None:
        await self.trade(
            f"Trade Opened: {symbol}",
            f"Side: {side}\nSize: {size:.4f}\nPrice: ${price:.2f}"
        )
    
    async def trade_closed(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
    ) -> None:
        level = AlertLevel.SUCCESS if pnl > 0 else AlertLevel.WARNING
        emoji = "🟢" if pnl > 0 else "🔴"
        await self.send(Alert(
            level,
            f"Trade Closed: {symbol} {emoji}",
            f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)"
        ))
    
    async def drawdown_warning(self, drawdown: float) -> None:
        await self.warning(
            "Drawdown Alert",
            f"Current drawdown: {drawdown:.1%}\nConsider reducing positions."
        )
    
    async def system_error(self, error: str) -> None:
        await self.critical(
            "System Error",
            f"Error: {error}\nTrading may be affected."
        )


# Singleton instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance.
    
    Returns:
        AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
