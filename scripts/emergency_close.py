#!/usr/bin/env python
"""Emergency position closing script."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.data.binance_client import BinanceClient


async def close_all_positions(
    symbols: list[str] = None,
    testnet: bool = True,
    dry_run: bool = True,
) -> None:
    """Close all open positions.
    
    Args:
        symbols: Specific symbols to close (None = all)
        testnet: Whether to use testnet
        dry_run: If True, don't execute actual trades
    """
    logger.info("=" * 60)
    logger.info("EMERGENCY POSITION CLOSE")
    logger.info("=" * 60)
    
    if dry_run:
        logger.warning("DRY RUN MODE - No actual trades will be executed")
    
    client = BinanceClient(testnet=testnet)
    
    # Get all open positions
    try:
        account = client.client.futures_account()
        positions = account.get("positions", [])
    except Exception as e:
        logger.error(f"Failed to get account info: {e}")
        return
    
    # Filter to non-zero positions
    open_positions = []
    for pos in positions:
        amount = float(pos.get("positionAmt", 0))
        if abs(amount) > 0:
            symbol = pos.get("symbol")
            if symbols is None or symbol in symbols:
                open_positions.append({
                    "symbol": symbol,
                    "amount": amount,
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedProfit", 0)),
                })
    
    if not open_positions:
        logger.info("No open positions found")
        return
    
    logger.info(f"Found {len(open_positions)} open positions:")
    
    for pos in open_positions:
        side = "LONG" if pos["amount"] > 0 else "SHORT"
        logger.info(
            f"  {pos['symbol']}: {side} {abs(pos['amount']):.4f} "
            f"@ ${pos['entry_price']:.2f} "
            f"(PnL: ${pos['unrealized_pnl']:.2f})"
        )
    
    if dry_run:
        logger.info("DRY RUN - Would close the above positions")
        return
    
    # Confirm
    print("\n⚠️  WARNING: This will close ALL positions above!")
    confirm = input("Type 'CONFIRM' to proceed: ")
    
    if confirm != "CONFIRM":
        logger.info("Cancelled")
        return
    
    # Close each position
    for pos in open_positions:
        symbol = pos["symbol"]
        amount = pos["amount"]
        
        # Opposite side to close
        side = "SELL" if amount > 0 else "BUY"
        quantity = abs(amount)
        
        try:
            logger.info(f"Closing {symbol}: {side} {quantity}")
            
            order = client.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                reduceOnly=True,
            )
            
            logger.success(f"Closed {symbol}: Order ID {order.get('orderId')}")
            
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
    
    logger.info("=" * 60)
    logger.info("Position close completed")


async def cancel_all_orders(
    symbols: list[str] = None,
    testnet: bool = True,
    dry_run: bool = True,
) -> None:
    """Cancel all open orders.
    
    Args:
        symbols: Specific symbols to cancel (None = all)
        testnet: Whether to use testnet
        dry_run: If True, don't execute actual cancellations
    """
    logger.info("Cancelling all open orders...")
    
    client = BinanceClient(testnet=testnet)
    
    try:
        if symbols:
            for symbol in symbols:
                orders = client.client.futures_get_open_orders(symbol=symbol)
                for order in orders:
                    if not dry_run:
                        client.client.futures_cancel_order(
                            symbol=symbol,
                            orderId=order["orderId"],
                        )
                    logger.info(f"{'Would cancel' if dry_run else 'Cancelled'}: {symbol} order {order['orderId']}")
        else:
            if not dry_run:
                client.client.futures_cancel_all_open_orders()
            logger.info(f"{'Would cancel' if dry_run else 'Cancelled'} all orders")
            
    except Exception as e:
        logger.error(f"Failed to cancel orders: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Emergency position management")
    parser.add_argument(
        "--action",
        type=str,
        choices=["close", "cancel", "both"],
        default="close",
        help="Action to perform",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Specific symbols (default: all)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute real trades (default: dry run)",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        default=True,
        help="Use testnet (default: True)",
    )
    args = parser.parse_args()
    
    dry_run = not args.live
    
    if args.action in ("close", "both"):
        asyncio.run(close_all_positions(
            symbols=args.symbols,
            testnet=args.testnet,
            dry_run=dry_run,
        ))
    
    if args.action in ("cancel", "both"):
        asyncio.run(cancel_all_orders(
            symbols=args.symbols,
            testnet=args.testnet,
            dry_run=dry_run,
        ))


if __name__ == "__main__":
    main()
