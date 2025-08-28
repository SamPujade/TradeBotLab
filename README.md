# Binance Trading Bot Simulator

This project provides a framework for creating and simulating crypto trading bots using the Binance API. It allows for the evaluation and comparison of different bot strategies under simulated scenarios (e.g., 100 USDT invested, 1-minute intervals). This is purely for simulation purposes; no real trading is involved.

## Project Structure

*   `pyproject.toml`: Project metadata and dependencies.
*   `evaluate.py`: Script to run simulations and evaluate bot performance.
*   `generate_dataset.py`: Script to generate datasets for model training.
*   `rnn_test.py`: Script for testing RNN models.
*   `train.py`: Script for training models.
*   `src/`: Contains core application logic.
    *   `bot.py`: Defines base `Bot` class and various bot implementations (e.g., `RandomBot`, `ProfitBot`, `RNNBot`).
    *   `dataset.py`: Handles data loading and processing for models.
    *   `trader.py`: (Currently a placeholder) Intended for real-time trading logic.
    *   `utils.py`: Utility functions and enums (e.g., `Action`).
    *   `visualization.py`: Functions for plotting trade history.
    *   `models/`: Contains machine learning model definitions.
        *   `rnn.py`: Recurrent Neural Network model definition.

## Setup

1.  **Install dependencies**:
    This project uses `uv` for dependency management. Ensure you have `uv` installed.
    ```bash
    uv pip install --with-sources .
    ```
    Alternatively, you can use:
    ```bash
    uv pip install --sync .
    ```

2.  **Binance API Credentials**:
    Update `API_KEY` and `API_SECRET` in `.env` with your Binance API credentials.

## Usage

*   **Run Evaluation**:
    ```bash
    python evaluate.py
    ```
    This will run the defined bots against historical data and output their performance. Plots will be generated in the current directory.

*   **Train Models**:
    ```bash
    uv run train.py --model_type rnn
    ```
    Available model types are `rnn` and `rl`.

*   **Generate Dataset**:
    ```bash
    python generate_dataset.py
    ```

## Bots Implemented

*   **RandomBot**: Makes random buy/sell/wait decisions.
*   **ProfitBot**: Implements a rule-based strategy with target profit, stop loss, and missed profit logic.
*   **RNNBot**: Uses a Recurrent Neural Network to predict trading decision.
*   **RLBot**: Uses a Reinforcement Learning agent to predict trading decision.

## Future Improvements

*   Implement a proper `trader.py` for real-time trading.
*   Enhance dashboard output for `evaluate.py` (e.g., interactive plots, comprehensive summary).
*   Add more sophisticated trading strategies.
*   Improve documentation for training and dataset generation.