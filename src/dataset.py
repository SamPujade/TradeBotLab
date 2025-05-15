from binance.client import Client
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from typing import List

# --- CONFIGURATION ---
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
client = Client(API_KEY, API_SECRET)


# --- HELPERS ---
def get_interval_ms(interval):
    lookup = {
        Client.KLINE_INTERVAL_1MINUTE: 60_000,
        Client.KLINE_INTERVAL_3MINUTE: 180_000,
        Client.KLINE_INTERVAL_5MINUTE: 300_000,
        Client.KLINE_INTERVAL_15MINUTE: 900_000,
        Client.KLINE_INTERVAL_30MINUTE: 1_800_000,
        Client.KLINE_INTERVAL_1HOUR: 3_600_000,
        Client.KLINE_INTERVAL_2HOUR: 7_200_000,
        Client.KLINE_INTERVAL_4HOUR: 14_400_000,
        Client.KLINE_INTERVAL_6HOUR: 21_600_000,
        Client.KLINE_INTERVAL_8HOUR: 28_800_000,
        Client.KLINE_INTERVAL_12HOUR: 43_200_000,
        Client.KLINE_INTERVAL_1DAY: 86_400_000,
        Client.KLINE_INTERVAL_3DAY: 259_200_000,
        Client.KLINE_INTERVAL_1WEEK: 604_800_000,
        Client.KLINE_INTERVAL_1MONTH: 2_592_000_000,
    }
    return lookup[interval]


def date_to_millis(date_str):
    dt = datetime.strptime(date_str, "%d %b %Y")
    return int(dt.timestamp() * 1000)


def millis_to_datetime(ms):
    return datetime.utcfromtimestamp(ms / 1000.0)


def fetch_klines_chunk(symbol, interval, start_ts, end_ts, limit=1000):
    data = client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ts,
        endTime=end_ts,
        limit=limit,
    )
    return data


# --- MAIN DOWNLOAD LOOP ---
def download_and_save_klines(symbol, interval, start_str, end_str):
    start_ts = date_to_millis(start_str)
    end_ts = date_to_millis(end_str)
    interval_ms = get_interval_ms(interval)

    klines = []
    current_ts = start_ts

    while current_ts < end_ts:
        data = fetch_klines_chunk(symbol, interval, current_ts, end_ts)
        if not data:
            break
        klines.extend(data)

        last_open_time = data[-1][0]
        current_ts = last_open_time + interval_ms
        time.sleep(0.3)

        print(
            f"Fetched {len(klines)} rows so far. Last date: {millis_to_datetime(last_open_time)}"
        )

    # Convert to DataFrame
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.drop(columns=["ignore"], inplace=True)

    # Cast columns to float where needed
    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    df[float_cols] = df[float_cols].astype(float)

    output_dir = f"data/{symbol}/{interval}/"
    os.makedirs(output_dir, exist_ok=True)

    # Split and save by year
    df["year"] = df["open_time"].dt.year
    for year, group in df.groupby("year"):
        output_path = os.path.join(output_dir, f"{symbol}_{interval}_{year}.parquet")
        group.drop(columns="year").to_parquet(output_path, index=False)
        print(f"Saved {len(group)} rows to {output_path}")


def load_klines(file_paths: List[str]):
    df = pd.concat(
        [pd.read_parquet(path) for path in sorted(file_paths)], ignore_index=True
    )
    df = df.sort_values("open_time")
    klines = df[["open_time", "open", "high", "low", "close", "volume"]].values.tolist()

    return klines


# download_and_save_klines(symbol, interval, start_str, end_str)
