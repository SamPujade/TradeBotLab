from config.config import Config  # Import Config for COMMISSION_RATE
from src.utils import Action

# from models.rnn import KlineDataset, RNN # This will be specific to RNNBot, not base Bot


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
        # COMMISSION is used here, it needs to be imported or passed
        # For now, I'll assume it will be handled by a config or parameter.
        # I'll use a placeholder for now, and address it with config later.
        if action == Action.BUY and self.balance > 0:
            self.position = (self.balance / price) * (1 - Config.COMMISSION_RATE)
            self.entry_price = price
            self.exit_price = None
            self.balance = 0
        elif action == Action.SELL and self.position > 0:
            self.balance = self.position * price * (1 - Config.COMMISSION_RATE)
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
