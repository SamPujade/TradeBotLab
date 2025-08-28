from src.bots.base_bot import Bot
from src.utils import Action


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
