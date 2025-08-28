from config.config import Config
from src.bots.base_bot import Bot
from src.models.rnn import RNN, KlineDataset
from src.utils import Action


class RNNBot(Bot):
    def __init__(
        self,
        balance,
        sequence_length=Config.SEQUENCE_LENGTH,
        model_path="rnn_weights.pth",
    ):
        self.name = "rnn_bot"
        self.balance = balance
        self.position = 0
        self.sequence_length = sequence_length
        self.trade_history = []
        self.model = RNN(
            input_size=5,  # 5 features: Open, High, Low, Close, Volume
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_NUM_LAYERS,
            output_size=3,  # 3 actions: BUY, SELL, WAIT
        )

        if model_path:
            self.model.load_weights(model_path)
        else:
            print("Warning: No model weights loaded!")

    def action(self, klines):
        # Ensure enough klines for a sequence
        if len(klines) < self.sequence_length:
            return Action.WAIT

        # Create a KlineDataset from the last 'sequence_length' klines
        prediction_klines = klines[-self.sequence_length :]
        prediction_dataset = KlineDataset(
            prediction_klines, sequence_length=self.sequence_length
        )

        # Get the single sequence from the prediction_dataset
        input_sequence, _ = prediction_dataset[0]

        # Predict the action (0, 1, or 2)
        predicted_action_idx = self.model.predict(input_sequence.unsqueeze(0))

        if predicted_action_idx == 0:  # BUY
            return Action.BUY
        elif predicted_action_idx == 1:  # SELL
            return Action.SELL
        else:  # WAIT
            return Action.WAIT
