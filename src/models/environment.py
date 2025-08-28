import numpy as np

from config.config import Config


class TradingEnvironment:
    def __init__(
        self,
        klines,
        initial_balance=Config.INITIAL_BALANCE,
        commission_rate=Config.COMMISSION_RATE,
    ):
        self.klines = klines
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # BTC held
        self.history = []  # To store (balance, position, price, action) at each step

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.history = []
        return self._get_state()

    def _get_state(self):
        # State will be the last N_KLINES as features
        # For simplicity, let's just use the last kline for now (Open, High, Low, Close, Volume)
        # This needs to be consistent with the input size of the RL agent.
        if self.current_step < Config.N_KLINES - 1:
            # Not enough klines to form a full state yet
            # Pad with zeros or handle appropriately (e.g., return initial state)
            state_data = np.zeros((Config.N_KLINES, 5))  # Assuming 5 features per kline
        else:
            start_index = self.current_step - Config.N_KLINES + 1
            end_index = self.current_step + 1
            state_klines = self.klines[start_index:end_index]
            state_data = np.array(
                [
                    [
                        float(kline[1]),
                        float(kline[2]),
                        float(kline[3]),
                        float(kline[4]),
                        float(kline[5]),
                    ]
                    for kline in state_klines
                ]
            )

        # Normalize state data if necessary (e.g., min-max or z-score scaling)
        # For now, return raw data.
        return state_data.flatten()  # Flatten for a single vector state

    def step(self, action):
        # Actions: 0 = BUY, 1 = SELL, 2 = HOLD (WAIT)
        current_kline = self.klines[self.current_step]
        current_price = float(current_kline[4])  # Close price

        reward = 0
        done = False

        if action == 0:  # BUY
            if self.balance > 0:
                self.position = (self.balance / current_price) * (
                    1 - self.commission_rate
                )
                self.balance = 0
                reward = 1  # Simple reward for taking action, can be refined
            else:
                reward = (
                    -0.1
                )  # Penalty for invalid action (trying to buy without balance)
        elif action == 1:  # SELL
            if self.position > 0:
                self.balance = (
                    self.position * current_price * (1 - self.commission_rate)
                )
                self.position = 0
                reward = 1  # Simple reward for taking action, can be refined
            else:
                reward = (
                    -0.1
                )  # Penalty for invalid action (trying to sell without position)
        elif action == 2:  # HOLD
            reward = 0.01  # Small reward for holding, encourages not losing money

        self.history.append((self.balance, self.position, current_price, action))

        self.current_step += 1
        if self.current_step >= len(self.klines):
            done = True
            # Calculate final portfolio value
            final_value = self.balance + (self.position * current_price)
            reward += (
                (final_value - self.initial_balance) / self.initial_balance * 100
            )  # Percentage profit/loss as reward

        next_state = self._get_state()
        return next_state, reward, done, {}  # obs, reward, done, info
