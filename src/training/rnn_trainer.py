import datetime
import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config.config import Config
from models.rnn import RNN, KlineDataset


class RNNTrainer:
    def __init__(self):
        pass

    def train_model(
        self,
        klines,
        symbol="BTCUSDT",  # This can be passed from outside if needed
        interval=Config.INTERVAL,  # Use Config.INTERVAL
        limit=None,  # Not used when klines are passed directly
        sequence_length=Config.SEQUENCE_LENGTH,
        epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        batch_size=Config.BATCH_SIZE,
        weights_folder=Config.WEIGHTS_FOLDER,
    ):
        """Trains the RNN model with the given parameters."""

        # Load klines data
        # If klines are already provided, no need to fetch or load from file
        # The original train.py had two paths for klines: direct fetch or load from file.
        # For a trainer, it's better to receive klines as an argument.

        # Split into train/test
        usable_length = len(klines) - sequence_length - Config.PREDICT_STEPS
        split_index = int(usable_length * (1 - Config.TEST_RATIO))
        train_klines = klines[: split_index + sequence_length]
        test_klines = klines[split_index:]

        # Create datasets
        train_dataset = KlineDataset(
            train_klines,
            sequence_length=sequence_length,
            predict_steps=Config.PREDICT_STEPS,
        )
        test_dataset = KlineDataset(
            test_klines,
            sequence_length=sequence_length,
            predict_steps=Config.PREDICT_STEPS,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Initialize and train model
        model = RNN(
            input_size=5,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_NUM_LAYERS,
            output_size=3,  # Output is 3 for BUY/SELL/WAIT
        )
        losses = model.train_model(
            train_dataloader, num_epochs=epochs, learning_rate=learning_rate
        )

        # Generate the training folder if it doesn't exist
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        training_folder = os.path.join(weights_folder, timestamp)
        if not os.path.exists(training_folder):
            os.makedirs(training_folder)

        # Save model weights
        model_path = os.path.join(training_folder, "btc_rnn_weights.pth")
        model.save_weights(model_path)
        print(f"Model trained and saved to {model_path}")

        # Save loss
        loss_path = os.path.join(training_folder, "btc_rnn_loss.png")
        self.plot_loss(losses, loss_path)
        return model_path

    def plot_loss(self, losses, loss_path):
        # Plot loss evolution
        plt.plot(losses)
        plt.title("Training Loss Evolution")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(loss_path)
        plt.close()

        print(f"Loss plot saved to {loss_path}")
