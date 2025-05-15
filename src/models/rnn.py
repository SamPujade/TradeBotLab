import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os

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
        target = self.data[
            idx
            + self.sequence_length : idx
            + self.sequence_length
            + self.predict_steps,
            3,
        ]  # Target closing price for next predict steps
        return sequence, target


class RNN(nn.Module):

    def __init__(
        self, input_size=5, hidden_size=64, num_layers=2, output_size=PREDICT_STEPS
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Predict next closing price
        return out

    def train_model(self, train_loader, num_epochs=20, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        losses = []
        for epoch in range(num_epochs):
            for sequences, targets in train_loader:
                sequences, targets = sequences.to(torch.float32), targets.to(
                    torch.float32
                )

                optimizer.zero_grad()
                outputs = self(sequences)
                targets = targets.view(outputs.shape)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

        return losses

    def predict(self, input_sequence):
        self.eval()
        with torch.no_grad():
            input_sequence = input_sequence.to(torch.float32)
            return self(input_sequence).cpu().numpy()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, model_path="rnn_weights.pth"):
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            print(f"Loaded weights from {model_path}")
        else:
            print(f"Weights file not found at {model_path}")
