import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.rnn import (
    KlineDataset,
    RNN,
)  # Make sure these are updated with PREDICT_STEPS support

PREDICT_STEPS = 10
SEQUENCE_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 1
TEST_RATIO = 0.1

if __name__ == "__main__":

    file_paths = glob.glob("data/klines/BTCUSDT/1m/BTCUSDT_1m_2024.parquet")
    df = pd.concat(
        [pd.read_parquet(path) for path in sorted(file_paths)], ignore_index=True
    )
    df = df.sort_values("open_time")
    klines = df[["open_time", "open", "high", "low", "close", "volume"]].values.tolist()

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
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize model
    model = RNN(input_size=5, hidden_size=64, num_layers=2, output_size=PREDICT_STEPS)
    # model.train_model(train_dataloader, num_epochs=EPOCHS)
    # model.save_weights("rnn_test.pth")
    model.load_weights("rnn_test.pth")

    model.eval()
    all_preds = []
    all_targets = []

    for sequence, target in test_dataloader:
        prediction = model.predict(sequence)
        prediction = test_dataset.denormalize(torch.tensor(prediction))
        target_denorm = test_dataset.denormalize(target)
        all_targets.append(target_denorm)
        all_preds.append(prediction)

    # Evaluate
    targets_np = torch.cat(all_targets, dim=0).numpy()
    preds_np = torch.cat(all_preds, dim=0).numpy()

    mse = mean_squared_error(targets_np, preds_np)
    mae = mean_absolute_error(targets_np, preds_np)

    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")

    # Predict multiple sequences
    test_inputs = torch.stack(
        [test_dataset[i * (SEQUENCE_LENGTH + PREDICT_STEPS)][0] for i in range(10)]
    )
    predicted_sequences = model.predict(test_inputs)  # Shape: (10, 10)
    predicted_sequences = test_dataset.denormalize(torch.tensor(predicted_sequences))
    predicted_flat = predicted_sequences.numpy().flatten()

    actual_sequences = []
    for i in range(10):
        actual = test_dataset[i * (SEQUENCE_LENGTH + PREDICT_STEPS)][1]  # Shape: (10,)
        actual_sequences.append(actual)
    actual_sequences = test_dataset.denormalize(torch.stack(actual_sequences))
    actual_flat = actual_sequences.numpy().flatten()

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(
        actual_flat,
        label=f"Actual {i}",
        color="blue",
        alpha=0.3,
    )
    plt.plot(
        predicted_flat,
        label=f"Predicted {i}",
        color="red",
        linestyle="dashed",
        alpha=0.3,
    )

    plt.title(f"Multi-step Prediction (next {PREDICT_STEPS} minutes)")
    plt.xlabel("Future Time Step")
    plt.ylabel("Denormalized Closing Price")
    plt.legend(["Actual", "Predicted"], loc="upper left")
    plt.tight_layout()
    plt.savefig("multi_step_prediction_plot_3.png")
    plt.close()
