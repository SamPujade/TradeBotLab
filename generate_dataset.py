"""
Example usage:

    python generate_dataset.py \
        --symbol BTCUSDT \
        --interval 1h \
        --start "2 Jan 2024" \
        --end "1 Jan 2025"
"""

import argparse
import sys
from binance.client import Client

from src.dataset import download_and_save_klines

INTERVAL_MAP = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "3m": Client.KLINE_INTERVAL_3MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "2h": Client.KLINE_INTERVAL_2HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "8h": Client.KLINE_INTERVAL_8HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "3d": Client.KLINE_INTERVAL_3DAY,
    "1w": Client.KLINE_INTERVAL_1WEEK,
    "1M": Client.KLINE_INTERVAL_1MONTH,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Binance kline data and save to Parquet."
    )
    parser.add_argument(
        "--symbol", type=str, required=True, help="Trading pair symbol (e.g., BTCUSDT)"
    )
    parser.add_argument(
        "--interval", type=str, required=True, help="Kline interval (e.g., 1m, 1h, 1d)"
    )
    parser.add_argument(
        "--start", type=str, required=True, help="Start date (e.g., '1 Jan 2022')"
    )
    parser.add_argument(
        "--end", type=str, required=True, help="End date (e.g., '1 Jan 2024')"
    )

    args = parser.parse_args()

    if args.interval not in INTERVAL_MAP:
        print(f"Error: Invalid interval '{args.interval}'.")
        print("Supported intervals:", ", ".join(INTERVAL_MAP.keys()))
        sys.exit(1)

    interval = INTERVAL_MAP[args.interval]

    # Call your dataset function
    download_and_save_klines(
        symbol=args.symbol, interval=interval, start_str=args.start, end_str=args.end
    )
