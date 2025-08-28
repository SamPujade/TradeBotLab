import random

from src.bots.base_bot import Bot
from src.utils import Action


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
