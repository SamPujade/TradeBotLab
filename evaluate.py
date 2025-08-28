import os

from binance.client import Client

from config.config import Config
from src.bots.profit_bot import ProfitBot
from src.bots.random_bot import RandomBot
from src.bots.rl_bot import RLBot  # Import RLBot
from src.bots.rnn_bot import RNNBot
from src.simulation.simulator import Simulator  # Import Simulator

# Run evaluation
if __name__ == "__main__":
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)

    # Fetch historical data using Binance API
    client = Client(Config.BINANCE_API_KEY, Config.BINANCE_API_SECRET, testnet=False)
    klines = client.get_klines(
        symbol="BTCUSDT",
        interval=Config.INTERVAL,
        limit=Config.N_KLINES + Config.N_EVALUATE,
    )

    bots = [
        RandomBot(balance=Config.INITIAL_BALANCE),
        ProfitBot(balance=Config.INITIAL_BALANCE),
        RNNBot(
            balance=Config.INITIAL_BALANCE,
            model_path=Config.RNN_MODEL_PATH,
        ),
        RLBot(
            balance=Config.INITIAL_BALANCE,
            state_size=Config.N_KLINES * 5,  # Needs to match environment's state size
            action_size=3,  # BUY, SELL, HOLD
            seed=42,  # Or make configurable
        ),
    ]

    simulator = Simulator(klines)
    results = simulator.run_simulation(bots)
