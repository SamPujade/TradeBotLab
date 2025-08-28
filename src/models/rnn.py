import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset

from config.config import Config


class KlineDataset(Dataset):
    def __init__(
        self,
        klines,
        predict_steps=Config.PREDICT_STEPS,
        sequence_length=Config.SEQUENCE_LENGTH,
        mean=None,
        std=None,
    ):
        self.predict_steps = predict_steps
        self.sequence_length = sequence_length
        self.unnormalized_data = torch.tensor(
            [self.preprocess_kline(kline) for kline in klines], dtype=torch.float32
        )

        if mean is None or std is None:
            self.mean = self.unnormalized_data.mean(dim=0)
            self.std = self.unnormalized_data.std(dim=0) + 1e-8
        else:
            # validation/test set
            self.mean = mean
            self.std = std
        self.data = (self.unnormalized_data - self.mean) / self.std  # normalize

    def preprocess_kline(self, kline):
        return [
            float(kline[1]),  # Open
            float(kline[2]),  # High
            float(kline[3]),  # Low
            float(kline[4]),  # Close
            float(kline[5]),  # Volume
        ]

    def __len__(self):
        return len(self.data) - self.sequence_length - self.predict_steps

    def __getitem__(self, idx):
        sequence = self.data[idx : idx + self.sequence_length]

        # Determine the action based on future price movement
        current_price = self.unnormalized_data[
            idx + self.sequence_length - 1, 3
        ]  # Last closing price in the sequence
        future_price = self.unnormalized_data[
            idx + self.sequence_length + self.predict_steps - 1, 3
        ]  # Closing price after PREDICT_STEPS

        price_change = (future_price - current_price) / current_price

        if price_change > Config.ACTION_PREDICTION_THRESHOLD:
            target_action = 0  # BUY
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

        weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=weights)  # for classification

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Output logits for actions
        return out

    def train_model(self, train_loader, num_epochs=20, learning_rate=0.001):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        losses = []
        n_batches = len(train_loader)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(torch.float32)
                targets = targets.to(torch.long)  # long for CrossEntropyLoss

                optimizer.zero_grad()
                outputs = self(sequences)
                # targets as (batch_size) and outputs as (batch_size, num_classes)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print(
                    f"\rEpoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{n_batches}]",
                    end="",
                )

            avg_epoch_loss = running_loss / n_batches
            losses.append(avg_epoch_loss)
            print(
                f"\rEpoch [{epoch + 1}/{num_epochs}] --- Average Loss: {avg_epoch_loss:.6f}         "
            )

        return losses

    def test_model(self, test_loader):
        self.eval()

        total_loss = 0
        all_preds = []
        all_labels = []
        class_names = ["BUY", "SELL", "WAIT"]

        with torch.no_grad():
            for sequences, labels in test_loader:
                outputs = self(sequences)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0
        )

        return avg_loss, accuracy, report, all_labels, all_preds

    def predict(self, input_sequence):
        self.eval()
        with torch.no_grad():
            input_sequence = input_sequence.to(torch.float32)
            outputs = self(input_sequence)
            return torch.argmax(outputs, dim=1).item()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, model_path="rnn_weights.pth"):
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            print(f"Loaded weights from {model_path}")
        else:
            print(f"Weights file not found at {model_path}")
