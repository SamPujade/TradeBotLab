from config.config import Config
from src.bots.base_bot import Bot
from src.models.environment import TradingEnvironment
from src.models.rl_agent import Agent
from src.utils import Action


class RLBot(Bot):
    def __init__(self, balance, state_size, action_size, seed=42):
        super().__init__(balance)
        self.name = "rl_bot"
        self.agent = Agent(state_size, action_size, seed)
        self.current_state = None  # To store the state from the environment

    def action(self, klines):
        # In the context of a bot's action method, klines represents the historical data
        # available up to the current point.
        # We need to extract the state that the RL agent expects.
        # The TradingEnvironment's _get_state method expects a list of klines,
        # so we can use that to get the state for the agent.

        # Create a dummy environment to get the state from the klines
        # This is not ideal as it re-initializes the environment, but for now it works
        # A better approach would be for the simulator to provide the state directly.
        # For now, let's assume the klines passed to `action` are sufficient to
        # construct the state.

        # The state is derived from the klines. The TradingEnvironment expects
        # a list of klines and processes it to a flat numpy array.
        # We need to ensure the klines here are sufficient to form a state
        # of `Config.N_KLINES` length.

        # Ensure klines has enough data for the state
        if len(klines) < Config.N_KLINES:
            # Not enough data to form a state, return WAIT for now
            return Action.WAIT

        # Create a temporary environment instance just to get the state representation
        # from the provided klines. This is a bit of a hack and should be improved
        # when the Simulator is implemented to provide the state directly.
        temp_env = TradingEnvironment(klines)
        temp_env.current_step = len(klines) - 1  # Set current step to the last kline
        state = temp_env._get_state()

        # The agent acts on the state
        action_idx = self.agent.act(state)

        if action_idx == 0:
            return Action.BUY
        elif action_idx == 1:
            return Action.SELL
        else:
            return Action.WAIT
