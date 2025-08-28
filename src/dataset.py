import math
import os
import time
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from binance.client import Client

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


def generate_dummy_pattern_data(
    symbol,
    interval,
    start_price=65000,
    increment=50,
    steps_per_phase=50,
    total_steps_target=10000,
    start_date_str="1 Jan 2022",
    interval_hours=1,
):
    """
    Generates a large dummy dataset by repeating an up-then-down price pattern.

    Args:
        start_price (float): The starting price for the pattern.
        increment (float): The amount to change the price at each step.
        steps_per_phase (int): The number of steps in the upward and downward phases.
        total_steps_target (int): The total number of klines to generate.
        start_date_str (str): The starting date for the dataset.
        interval_hours (int): The interval in hours between each kline.

    Returns:
        pd.DataFrame: A DataFrame with the generated kline data.
    """
    klines = []
    current_price = start_price
    start_time = datetime.strptime(start_date_str, "%d %b %Y")
    interval_delta = timedelta(hours=interval_hours)
    interval_ms = interval_delta.total_seconds() * 1000

    # Calculate how many times the full pattern needs to repeat
    pattern_length = 2 * steps_per_phase
    num_repetitions = math.ceil(total_steps_target / pattern_length)

    global_step_counter = 0

    print(
        f"Generating {total_steps_target} steps by repeating a {pattern_length}-step pattern..."
    )

    # --- New outer loop to repeat the pattern ---
    for _ in range(num_repetitions):
        # Phase 1: Increasing Price
        for _ in range(steps_per_phase):
            open_time = start_time + (global_step_counter * interval_delta)
            open_time_ms = int(open_time.timestamp() * 1000)

            open_price = current_price
            close_price = open_price + increment
            high_price = close_price + np.random.uniform(5, 10)
            low_price = open_price - np.random.uniform(5, 10)

            # Append kline data...
            klines.append(
                [
                    open_time_ms,
                    f"{open_price:.2f}",
                    f"{high_price:.2f}",
                    f"{low_price:.2f}",
                    f"{close_price:.2f}",
                    f"{np.random.uniform(10, 50):.8f}",
                    open_time_ms + interval_ms - 1,
                    f"{(np.random.uniform(10, 50) * current_price):.8f}",
                    np.random.randint(500, 1500),
                    f"{(np.random.uniform(10, 50) * 0.55):.8f}",
                    f"{(np.random.uniform(10, 50) * 0.55 * current_price):.8f}",
                    "0",
                ]
            )
            current_price = close_price
            global_step_counter += 1

        # Phase 2: Decreasing Price
        for _ in range(steps_per_phase):
            open_time = start_time + (global_step_counter * interval_delta)
            open_time_ms = int(open_time.timestamp() * 1000)

            open_price = current_price
            close_price = open_price - increment
            high_price = open_price + np.random.uniform(5, 10)
            low_price = close_price - np.random.uniform(5, 10)

            # Append kline data...
            klines.append(
                [
                    open_time_ms,
                    f"{open_price:.2f}",
                    f"{high_price:.2f}",
                    f"{low_price:.2f}",
                    f"{close_price:.2f}",
                    f"{np.random.uniform(10, 50):.8f}",
                    open_time_ms + interval_ms - 1,
                    f"{(np.random.uniform(10, 50) * current_price):.8f}",
                    np.random.randint(500, 1500),
                    f"{(np.random.uniform(10, 50) * 0.45):.8f}",
                    f"{(np.random.uniform(10, 50) * 0.45 * current_price):.8f}",
                    "0",
                ]
            )
            current_price = close_price
            global_step_counter += 1

    # Truncate the list to the exact number of steps required
    final_klines = klines[:total_steps_target]

    # --- Convert to DataFrame (same as before) ---
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
    df = pd.DataFrame(final_klines, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.drop(columns=["ignore"], inplace=True)
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
    df["number_of_trades"] = df["number_of_trades"].astype(int)

    # 2. Save it to a Parquet file, mimicking your saving logic
    output_dir = f"data/klines/{symbol}/{interval}/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{symbol}_{interval}_dummy.parquet")

    df.to_parquet(output_path, index=False)

    print("\n" + "=" * 50)
    print("✅ Successfully generated and saved dummy dataset to:")
    print(f"{output_path}")
    print("=" * 50)
    print("\nFirst 5 rows of the generated data:")
    print(df.head())
    print("\nLast 5 rows of the generated data:")
    print(df.tail())


# download_and_save_klines(symbol, interval, start_str, end_str)
