import torch
from binance.client import Client
import datetime
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.rnn import RNN, KlineDataset
from dataset import load_klines


DATA_PATH = "data/BTCUSDT/1m/BTCUSDT_1m_2024.parquet"
PREDICT_STEPS = 10
SEQUENCE_LENGTH = 100
BATCH_SIZE = 16
EPOCHS = 10
TEST_RATIO = 0.1

# Training parameters
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE
learning_rate = 0.001
batch_size = 4
weights_folder = "data/weights/"  # the folder where we save the weights.


def train_model(
    client,
    symbol,
    interval,
    limit,
    sequence_length,
    epochs,
    learning_rate,
    batch_size,
    weights_folder="data/weights/",
):
    """Trains the RNN model with the given parameters."""

    # Fetch klines data
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    # Generate timestamped folder name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    training_folder = os.path.join(weights_folder, timestamp)

    # Create the training folder if it doesn't exist
    if not os.path.exists(training_folder):
        os.makedirs(training_folder)

    model_path = os.path.join(training_folder, "btc_rnn_weights.pth")
    loss_path = os.path.join(training_folder, "btc_rnn_loss.png")

    # Initialize and train the model
    model = RNN()
    losses = model.train_model(
        klines, sequence_length, epochs, learning_rate, batch_size, model_path
    )

    print(f"Model trained and saved to {model_path}")

    plot_loss(losses, loss_path)


def plot_loss(losses, loss_path):
    # Plot loss evolution
    plt.plot(losses)
    plt.title("Training Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(loss_path)
    plt.close()

    print(f"Loss plot saved to {loss_path}")


if __name__ == "__main__":

    # Load klines data
    file_path = os.path.abspath(DATA_PATH)
    klines = load_klines([file_path])

    # Split into train/test
    usable_length = len(klines) - SEQUENCE_LENGTH - PREDICT_STEPS
    split_index = int(usable_length * (1 - TEST_RATIO))
    train_klines = klines[: split_index + SEQUENCE_LENGTH]
    test_klines = klines[split_index:]

    # Create datasets
    train_dataset = KlineDataset(
        train_klines, sequence_length=SEQUENCE_LENGTH, predict_steps=PREDICT_STEPS
    )
    test_dataset = KlineDataset(
        test_klines, sequence_length=SEQUENCE_LENGTH, predict_steps=PREDICT_STEPS
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize and train model
    model = RNN()
    losses = model.train_model(train_dataloader, num_epochs=EPOCHS)

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
    plot_loss(losses, loss_path)
