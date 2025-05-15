from binance.client import Client
import pandas as pd
import time

from src.bot import RandomBot, ProfitBot, RNNBot
from src.visualization import plot_trades

# Binance API credentials
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"

BALANCE = 100
N_KLINES = 9
N_EVALUATE = 900


# Fetch historical data using Binance API
def fetch_binance_data(
    symbol="BTCUSDT",
    interval=Client.KLINE_INTERVAL_1MINUTE,
    limit=N_KLINES + N_EVALUATE,
):
    client = Client(API_KEY, API_SECRET, testnet=False)
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    return klines


# Run evaluation
if __name__ == "__main__":
    klines = fetch_binance_data()
    bots = [
        RandomBot(balance=BALANCE),
        ProfitBot(balance=BALANCE),
        RNNBot(
            balance=BALANCE,
            model_path="data/weights/20250326_112452/btc_rnn_weigths.pth",
        ),
    ]

    for bot in bots:
        print(f"BOT {bot.name}")
        final_balance, trade_history = bot.evaluate(klines, N_EVALUATE)
        print(f"Final balance: {final_balance} USDT")
        plot_trades(
            klines,
            trade_history,
            starting_index=N_KLINES,
            filename=bot.name + "_evaluate_plot.png",
        )
        print("-------")
