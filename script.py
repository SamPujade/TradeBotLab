from binance.client import Client
import pandas as pd
import time

# Binance API credentials
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Trading parameters
symbol = "BTCUSDT"
interval = client.KLINE_INTERVAL_1MINUTE  # 1-minute candles
quantity = 0.001  # Adjust based on your balance
short_window = 5  # Short moving average period
long_window = 20  # Long moving average period


# Function to fetch historical data
def get_historical_data(symbol, interval, limit=50):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(
        klines,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["close"] = df["close"].astype(float)
    return df[["time", "close"]]


# # Function to calculate moving averages
# def calculate_sma(data, window):
#     return data["close"].rolling(window=window).mean()


# # Function to place a market order
# def place_order(order_type, quantity):
#     try:
#         order = client.order_market(symbol=symbol, side=order_type, quantity=quantity)
#         print(f"Order placed: {order_type} {quantity} {symbol}")
#     except Exception as e:
#         print(f"Error placing order: {e}")


# # Trading loop
# def trading_bot():
#     position = None  # "LONG" if we own BTC, None if not
#     while True:
#         df = get_historical_data(symbol, interval)
#         df["SMA_short"] = calculate_sma(df, short_window)
#         df["SMA_long"] = calculate_sma(df, long_window)

#         if df["SMA_short"].iloc[-1] > df["SMA_long"].iloc[-1] and position is None:
#             print("Buying BTC...")
#             place_order("BUY", quantity)
#             position = "LONG"

#         elif df["SMA_short"].iloc[-1] < df["SMA_long"].iloc[-1] and position == "LONG":
#             print("Selling BTC...")
#             place_order("SELL", quantity)
#             position = None

#         time.sleep(60)  # Wait for the next candle


# Run the bot
if __name__ == "__main__":
    df = get_historical_data(symbol, interval)
    print(df)
#     trading_bot()
