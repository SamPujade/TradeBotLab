import random
import numpy as np

from src.utils import Action
from models.rnn import KlineDataset, RNN

COMMISSION = 0.0015


class Bot:

    def __init__(self, balance):
        self.balance = balance  # Starting balance in USDT
        self.position = 0  # BTC held
        self.entry_price = None
        self.exit_price = None
        self.trade_history = []

    def action(self, klines=None):
        raise NotImplementedError("Subclasses must implement this method")

    def execute_trade(self, action, price):
        if action == Action.BUY and self.balance > 0:
            self.position = (self.balance / price) * (1 - COMMISSION)
            self.entry_price = price
            self.exit_price = None
            self.balance = 0
        elif action == Action.SELL and self.position > 0:
            self.balance = self.position * price * (1 - COMMISSION)
            self.position = 0
            self.exit_price = price
            self.entry_price = None

    def evaluate(self, klines, k):
        starting_index = len(klines) - k
        prices = [float(candle[4]) for candle in klines]

        for i in range(k):
            price = prices[starting_index + i]
            action = self.action(klines[i : starting_index + i + 1])
            self.execute_trade(action, price)
            self.trade_history.append((action, price))

        final_balance = self.balance + (self.position * prices[-1])
        return final_balance, self.trade_history


class RandomBot(Bot):

    def __init__(self, balance):
        super().__init__(balance)
        self.name = "random_bot"

    def action(self, klines=None):
        if self.position == 0:
            weights = [1, 0, 99]
        else:
            weights = [0, 1, 99]

        actions = list(Action)
        return random.choices(actions, weights=weights, k=1)[0]


class ProfitBot(Bot):

    def __init__(self, balance):
        super().__init__(balance)
        self.name = "profit_bot"
        self.stop_loss_offset = 0.005  # 10% below buy price
        self.missed_profit_offset = 0.005  # 10% above sell price
        self.buy_offset = 0.001  # 1% below selling price
        self.sell_offset = 0.002  # 2% above buying price.
        self.target_buy_price = None
        self.target_sell_price = None

    def action(self, klines=None):
        prices = [float(candle[4]) for candle in klines]
        price = prices[-1]

        # buy at start or when target buy price is reached
        if not self.position and (
            not self.target_buy_price or price <= self.target_buy_price
        ):
            self.target_sell_price = price * (1 + self.sell_offset)
            return Action.BUY

        # profit target is reached
        if self.position and price >= self.target_sell_price:
            self.target_buy_price = price * (1 - self.buy_offset)
            return Action.SELL

        # stop loss
        if self.entry_price and price <= self.entry_price * (1 - self.stop_loss_offset):
            self.target_buy_price = price * (1 - self.buy_offset)
            return Action.SELL

        # missed profit
        if self.exit_price and price >= self.exit_price * (
            1 + self.missed_profit_offset
        ):
            self.target_sell_price = price * (1 + self.sell_offset)
            return Action.BUY

        return Action.WAIT


class RNNBot(Bot):
    def __init__(self, balance, sequence_length=100, model_path="rnn_weights.pth"):
        self.name = "rnn_bot"
        self.balance = balance
        self.position = 0
        self.sequence_length = sequence_length
        self.trade_history = []
        self.model = RNN()

        if model_path:
            self.model.load_weights(model_path)
        else:
            print("Warning: No model weights loaded!")

    def action(self, klines):
        latest_price = klines[-1][4]

        dataset = KlineDataset(klines, sequence_length=self.sequence_length)
        last_sequence = dataset[-self.sequence_length :]
        predicted_price = self.model.predict(last_sequence.unsqueeze(0)).item()

        if predicted_price > latest_price * (1 + COMMISSION):
            if self.position:
                return Action.WAIT
            else:
                return Action.BUY
        elif predicted_price < latest_price * (1 - COMMISSION):
            if self.position:
                return Action.SELL
            else:
                return Action.WAIT

        return Action.WAIT
