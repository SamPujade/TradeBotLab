import argparse  # Import argparse
import os

from config.config import Config
from src.dataset import load_klines
from src.training.rl_trainer import RLTrainer  # Import RLTrainer
from src.training.rnn_trainer import RNNTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train different types of trading bot models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["rnn", "rl"],
        help="Type of model to train (rnn or rl)",
    )
    args = parser.parse_args()

    # Load klines data
    file_path = os.path.abspath(Config.DATA_PATH)
    klines = load_klines([file_path])

    if args.model_type == "rnn":
        print("Starting RNN model training...")
        rnn_trainer = RNNTrainer()
        model_path = rnn_trainer.train_model(klines)
        print(f"RNN Model training complete. Model saved to: {model_path}")
    elif args.model_type == "rl":
        print("Starting RL model training...")
        rl_trainer = RLTrainer()
        model_path = rl_trainer.train_agent(klines)
        print(f"RL Model training complete. Model saved to: {model_path}")
    else:
        print(f"Unknown model type: {args.model_type}")
