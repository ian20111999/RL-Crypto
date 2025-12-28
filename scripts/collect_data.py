"""Data collection script for RL-Crypto."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.data import DataCollector


def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description="Collect historical data from Binance")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--days", type=int, default=None, help="Number of days to fetch")
    parser.add_argument("--interval", type=str, default=None, help="Kline interval (1m, 5m, etc)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to fetch")
    parser.add_argument("--update", action="store_true", help="Update existing data")
    parser.add_argument("--info", action="store_true", help="Show data info only")
    args = parser.parse_args()

    # Setup logging
    logger.add(
        "logs/data_collection_{time}.log",
        rotation="100 MB",
        level="INFO",
    )

    logger.info("=" * 60)
    logger.info("RL-Crypto Data Collection")
    logger.info("=" * 60)

    collector = DataCollector(args.config)

    if args.info:
        # Show data info
        info = collector.get_data_info()
        if info.empty:
            logger.info("No data files found")
        else:
            print("\nData Files:")
            print(info.to_string(index=False))
        return

    if args.update:
        # Update existing data
        logger.info("Updating existing data...")
        collector.update_data(interval=args.interval)
    else:
        # Collect new data
        if args.symbols:
            # Collect specific symbols
            for symbol in args.symbols:
                collector.collect_symbol(
                    symbol=symbol,
                    interval=args.interval or collector.data_config["timeframe"],
                    days=args.days or collector.data_config["history_days"],
                )
        else:
            # Collect all configured symbols
            collector.collect_all(
                interval=args.interval,
                days=args.days,
            )

    # Show final info
    info = collector.get_data_info()
    print("\nData Files:")
    print(info.to_string(index=False))

    total_size = info["size_mb"].sum()
    logger.info(f"Total data size: {total_size:.2f} MB")
    logger.info("Data collection completed!")


if __name__ == "__main__":
    main()
