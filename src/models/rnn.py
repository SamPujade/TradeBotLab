import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from config.config import Config

PREDICT_STEPS = 10
SEQUENCE_LENGTH = 100


class KlineDataset(Dataset):
    def __init__(
        self, klines, predict_steps=PREDICT_STEPS, sequence_length=SEQUENCE_LENGTH
    ):
        self.predict_steps = predict_steps
        self.sequence_length = sequence_length
        self.data = [self.preprocess_kline(kline) for kline in klines]
        self.data = torch.tensor(self.data, dtype=torch.float32)

        # self.l2_norms = torch.norm(self.data, p=2, dim=1)
        # self.data = F.normalize(self.data, p=2, dim=1)

        # Feature-wise mean/std normalization
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0) + 1e-8  # avoid divide-by-zero
        self.data = (self.data - self.mean) / self.std

    def preprocess_kline(self, kline):
        return [
            float(kline[1]),  # Open
            float(kline[2]),  # High
            float(kline[3]),  # Low
            float(kline[4]),  # Close
            float(kline[5]),  # Volume
        ]

    def denormalize(self, values):
        # norms = self.l2_norms[: values.shape[0]].unsqueeze(1)  # shape: (N, 1)
        # return values * norms

        return values * self.std[3] + self.mean[3]

    def __len__(self):
        return len(self.data) - self.sequence_length - self.predict_steps

    def __getitem__(self, idx):
        sequence = self.data[idx : idx + self.sequence_length]
        # Determine the action based on future price movement
        current_price = self.data[
            idx + self.sequence_length - 1, 3
        ]  # Last closing price in the sequence
        future_price = self.data[
            idx + self.sequence_length + self.predict_steps - 1, 3
        ]  # Closing price after PREDICT_STEPS

        price_change = (future_price - current_price) / current_price

        if price_change > Config.ACTION_PREDICTION_THRESHOLD:
            target_action = 0  # BUY (arbitrary mapping for now)
        elif price_change < -Config.ACTION_PREDICTION_THRESHOLD:
            target_action = 1  # SELL
        else:
            target_action = 2  # WAIT

        return sequence, torch.tensor(target_action, dtype=torch.long)


class RNN(nn.Module):
    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        num_layers=2,
        output_size=3,  # Output size is 3 for BUY, SELL, WAIT
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(
            hidden_size, output_size
        )  # Output raw logits for classification

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Output logits for actions
        return out

    def train_model(self, train_loader, num_epochs=20, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        losses = []
        for epoch in range(num_epochs):
            for sequences, targets in train_loader:
                sequences = sequences.to(torch.float32)
                targets = targets.to(
                    torch.long
                )  # Targets should be long for CrossEntropyLoss

                optimizer.zero_grad()
                outputs = self(sequences)
                # CrossEntropyLoss expects targets as (batch_size) and outputs as (batch_size, num_classes)
                # No need for targets.view(outputs.shape) here
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

        return losses

    def predict(self, input_sequence):
        self.eval()
        with torch.no_grad():
            input_sequence = input_sequence.to(torch.float32)
            outputs = self(input_sequence)
            # Return the action with the highest probability (index)
            return torch.argmax(outputs, dim=1).item()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, model_path="rnn_weights.pth"):
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            print(f"Loaded weights from {model_path}")
        else:
            print(f"Weights file not found at {model_path}")
