import os

from binance.client import Client


class Config:
    # Binance API Credentials
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    # Simulation Configuration
    INITIAL_BALANCE = 100  # Starting balance in USDT
    N_KLINES = 9  # Number of klines to consider for bot action
    N_EVALUATE = 900  # Number of klines to evaluate over
    COMMISSION_RATE = 0.0015  # Trading commission rate
    INTERVAL = Client.KLINE_INTERVAL_1MINUTE  # Default kline interval for simulation
    PLOTS_DIR = "plots"  # Directory for saving plots

    # Training Configuration
    DATA_PATH = "data/klines/BTCUSDT/1m/BTCUSDT_1m_2024.parquet"
    PREDICT_STEPS = 10
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 128
    EPOCHS = 2
    TEST_RATIO = 0.1
    LEARNING_RATE = 0.001
    CLASS_WEIGHTS = [8, 8, 1.2]
    WEIGHTS_FOLDER = "data/weights/"
    ACTION_PREDICTION_THRESHOLD = 0.002  # 0.2% price change for action prediction

    # RNN Configuration:
    RNN_HIDDEN_SIZE = 64
    RNN_NUM_LAYERS = 2
    RNN_MODEL_PATH = "data/weights/20250828_222935/btc_rnn_weights.pth"

    # RL Configuration:
    RL_GAMMA = 0.99
    RL_EPS_START = 0.9
    RL_EPS_END = 0.05
    RL_EPS_DECAY = 1000
    RL_TAU = 0.005
    RL_LR = 1e-4
