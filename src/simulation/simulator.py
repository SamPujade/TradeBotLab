import os

import pandas as pd

from config.config import Config
from src.bots.base_bot import Bot
from src.visualization import plot_trades


class Simulator:
    def __init__(self, klines_data):
        self.klines_data = klines_data
        self.plots_dir = "plots"
        os.makedirs(self.plots_dir, exist_ok=True)

    def run_simulation(self, bots: list[Bot]):
        results = []
        for bot in bots:
            print(f"Evaluating BOT: {bot.name}")
            final_balance, trade_history = bot.evaluate(
                self.klines_data, Config.N_EVALUATE
            )

            plot_filename = os.path.join(self.plots_dir, f"{bot.name}_plot.png")
            plot_trades(
                self.klines_data,
                trade_history,
                starting_index=len(self.klines_data) - Config.N_EVALUATE,
                filename=plot_filename,
            )
            results.append(
                {
                    "Bot Name": bot.name,
                    "Final Balance (USDT)": f"{final_balance:.2f}",
                    "Plot File": plot_filename,
                }
            )
            print(f"  Final balance: {final_balance:.2f} USDT")
            print(f"  Plot saved to: {plot_filename}")
            print("-" * 30)

        print("\n--- Evaluation Summary ---")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        print("--------------------------")
        return results
