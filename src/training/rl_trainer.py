import datetime
import os
from collections import deque

import numpy as np
import torch

from config.config import Config
from src.models.environment import TradingEnvironment
from src.models.rl_agent import Agent


class RLTrainer:
    def __init__(self):
        pass

    def train_agent(
        self, klines, num_episodes=Config.EPOCHS, weights_folder=Config.WEIGHTS_FOLDER
    ):
        env = TradingEnvironment(klines)
        # Calculate state_size dynamically from the environment's _get_state method
        # Initialize a dummy environment to get the state size
        dummy_env = TradingEnvironment(
            klines[: Config.N_KLINES]
        )  # Pass minimal klines to initialize
        state_size = dummy_env._get_state().shape[0]  # Get the flattened state size

        action_size = 3  # BUY, SELL, HOLD
        agent = Agent(state_size, action_size, seed=42)

        scores = []
        scores_window = deque(maxlen=100)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        training_folder = os.path.join(weights_folder, timestamp)
        if not os.path.exists(training_folder):
            os.makedirs(training_folder)

        agent_weights_path = os.path.join(training_folder, "rl_agent_weights.pth")

        for i_episode in range(1, num_episodes + 1):
            state = env.reset()
            score = 0
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
            scores_window.append(score)
            scores.append(score)

            eps = max(
                Config.RL_EPS_END,
                Config.RL_EPS_START - (i_episode / Config.RL_EPS_DECAY),
            )
            agent.epsilon = eps

            print(
                f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}",
                end="",
            )
            if i_episode % 100 == 0:
                print(
                    f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}"
                )
                # Save agent weights periodically
                torch.save(
                    agent.qnetwork_local.state_dict(),
                    agent_weights_path.replace(".pth", f"_episode_{i_episode}.pth"),
                )

        torch.save(agent.qnetwork_local.state_dict(), agent_weights_path)
        print(f"\nRL Agent trained and saved to {agent_weights_path}")
        return agent_weights_path
