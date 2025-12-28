"""WebSocket client for real-time Binance data."""

import asyncio
import json
from typing import Callable, Optional
from datetime import datetime

from loguru import logger


class BinanceWebSocket:
    """WebSocket client for Binance Futures real-time data.
    
    Supports:
    - Kline/Candlestick streams
    - Mark price streams
    - Order book depth streams
    - Trade streams
    """
    
    BASE_URL = "wss://fstream.binance.com/ws"
    TESTNET_URL = "wss://fstream.binancefuture.com/ws"
    
    def __init__(
        self,
        symbols: list[str],
        testnet: bool = True,
        on_kline: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
        on_depth: Optional[Callable] = None,
        on_mark_price: Optional[Callable] = None,
    ):
        """Initialize WebSocket client.
        
        Args:
            symbols: List of symbols to subscribe
            testnet: Whether to use testnet
            on_kline: Callback for kline updates
            on_trade: Callback for trade updates
            on_depth: Callback for depth updates
            on_mark_price: Callback for mark price updates
        """
        self.symbols = [s.lower() for s in symbols]
        self.testnet = testnet
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        
        # Callbacks
        self.on_kline = on_kline
        self.on_trade = on_trade
        self.on_depth = on_depth
        self.on_mark_price = on_mark_price
        
        # State
        self.websocket = None
        self.is_running = False
        self._listen_task = None
        
        # Data buffers
        self.latest_prices: dict[str, float] = {}
        self.latest_klines: dict[str, dict] = {}
        
        logger.info(f"WebSocket client initialized for {len(symbols)} symbols (testnet={testnet})")

    def _build_streams(self) -> list[str]:
        """Build list of stream names to subscribe.
        
        Returns:
            List of stream names
        """
        streams = []
        for symbol in self.symbols:
            streams.append(f"{symbol}@kline_1h")  # 1 hour klines (matching trained model)
            streams.append(f"{symbol}@markPrice@1s")  # Mark price every second
            
            if self.on_trade:
                streams.append(f"{symbol}@aggTrade")  # Aggregated trades
            
            if self.on_depth:
                streams.append(f"{symbol}@depth5@100ms")  # Top 5 depth levels
        
        return streams

    async def connect(self) -> None:
        """Connect to WebSocket and start listening."""
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets package required. Install with: pip install websockets")
        
        streams = self._build_streams()
        stream_str = "/".join(streams)
        url = f"{self.base_url}/{stream_str}"
        
        logger.info(f"Connecting to {url[:100]}...")
        
        self.websocket = await websockets.connect(url)
        self.is_running = True
        
        logger.info("WebSocket connected successfully")
        
        # Start listening
        self._listen_task = asyncio.create_task(self._listen())

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self.is_running = False
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket disconnected")

    async def _listen(self) -> None:
        """Listen for incoming messages."""
        while self.is_running:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                await self._process_message(data)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.is_running:
                    await asyncio.sleep(1)
                    await self._reconnect()

    async def _reconnect(self) -> None:
        """Attempt to reconnect."""
        logger.info("Attempting to reconnect...")
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            await asyncio.sleep(5)

    async def _process_message(self, data: dict) -> None:
        """Process incoming message based on event type.
        
        Args:
            data: Message data
        """
        event_type = data.get("e")
        
        if event_type == "kline":
            await self._handle_kline(data)
        elif event_type == "markPriceUpdate":
            await self._handle_mark_price(data)
        elif event_type == "aggTrade":
            await self._handle_trade(data)
        elif event_type == "depthUpdate":
            await self._handle_depth(data)

    async def _handle_kline(self, data: dict) -> None:
        """Handle kline update.
        
        Args:
            data: Kline data
        """
        symbol = data["s"].upper()
        k = data["k"]
        
        kline = {
            "symbol": symbol,
            "interval": k["i"],
            "open_time": k["t"],
            "close_time": k["T"],
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "is_closed": k["x"],
        }
        
        self.latest_klines[symbol] = kline
        self.latest_prices[symbol] = kline["close"]
        
        if self.on_kline:
            await self._call_handler(self.on_kline, kline)

    async def _handle_mark_price(self, data: dict) -> None:
        """Handle mark price update.
        
        Args:
            data: Mark price data
        """
        symbol = data["s"].upper()
        
        mark_price = {
            "symbol": symbol,
            "mark_price": float(data["p"]),
            "index_price": float(data.get("i", 0)),
            "funding_rate": float(data.get("r", 0)),
            "next_funding_time": data.get("T", 0),
            "timestamp": data["E"],
        }
        
        self.latest_prices[symbol] = mark_price["mark_price"]
        
        if self.on_mark_price:
            await self._call_handler(self.on_mark_price, mark_price)

    async def _handle_trade(self, data: dict) -> None:
        """Handle trade update.
        
        Args:
            data: Trade data
        """
        symbol = data["s"].upper()
        
        trade = {
            "symbol": symbol,
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "is_buyer_maker": data["m"],
            "trade_time": data["T"],
        }
        
        if self.on_trade:
            await self._call_handler(self.on_trade, trade)

    async def _handle_depth(self, data: dict) -> None:
        """Handle depth update.
        
        Args:
            data: Depth data
        """
        symbol = data.get("s", "").upper()
        
        depth = {
            "symbol": symbol,
            "bids": [(float(p), float(q)) for p, q in data.get("b", [])],
            "asks": [(float(p), float(q)) for p, q in data.get("a", [])],
            "timestamp": data.get("E", 0),
        }
        
        if self.on_depth:
            await self._call_handler(self.on_depth, depth)

    async def _call_handler(self, handler: Callable, data: dict) -> None:
        """Call a handler function (sync or async).
        
        Args:
            handler: Handler function
            data: Data to pass to handler
        """
        if asyncio.iscoroutinefunction(handler):
            await handler(data)
        else:
            handler(data)

    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Latest price or None
        """
        return self.latest_prices.get(symbol.upper())

    def get_all_prices(self) -> dict[str, float]:
        """Get all latest prices.
        
        Returns:
            Dict of symbol to price
        """
        return self.latest_prices.copy()


async def example_usage():
    """Example usage of WebSocket client."""
    
    def on_kline(data):
        if data["is_closed"]:
            print(f"{data['symbol']} 1m candle closed: {data['close']}")
    
    def on_mark_price(data):
        print(f"{data['symbol']} mark: {data['mark_price']}, funding: {data['funding_rate']}")
    
    client = BinanceWebSocket(
        symbols=["BTCUSDT", "ETHUSDT"],
        testnet=True,
        on_kline=on_kline,
        on_mark_price=on_mark_price,
    )
    
    await client.connect()
    
    try:
        # Run for 60 seconds
        await asyncio.sleep(60)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(example_usage())
