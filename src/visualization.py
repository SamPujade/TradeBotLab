import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from src.utils import Action

OUTPUT_FOLDER = "data/outputs/"


def plot_trades(klines, trade_history, starting_index=0, filename="evaluate_plot.png"):
    prices = [float(candle[4]) for candle in klines]
    timestamps = [int(candle[0]) for candle in klines]

    # Convert to datetime
    dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]

    plt.figure(figsize=(14, 7))
    plt.plot(dates, prices, label="Price", color="blue")

    # Extract BUY/SELL points
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []

    for i, (action, price) in enumerate(trade_history):
        if action == Action.BUY:
            buy_dates.append(dates[starting_index + i])
            buy_prices.append(price)
        elif action == Action.SELL:
            sell_dates.append(dates[starting_index + i])
            sell_prices.append(price)

    plt.scatter(buy_dates, buy_prices, color="green", label="BUY", marker="^", s=100)
    plt.scatter(sell_dates, sell_prices, color="red", label="SELL", marker="v", s=100)

    plt.title("Price Evolution with Buy/Sell Orders")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Format X-axis: Show Hour:Minute
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # Show HH:MM

    # Set ticks every hour
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Every hour

    plt.gcf().autofmt_xdate()  # Beautify

    filepath = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(filepath)
    print(f"Plot saved as {filepath}")
    plt.close()
