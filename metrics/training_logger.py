import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

CONVERGENCE_WINDOW = 10          # rolling window for convergence check
CONVERGENCE_THRESHOLD = 0.90     # fraction of max reward seen so far


class EpisodeCSVLogger(BaseCallback):

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.rows_written = 0

        self._reward_history: list[float] = []
        self.convergence_episode: int | None = None 
        self.convergence_timestep: int | None = None

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep",
                "episode",
                "episode_reward",
                "episode_length",
                "episode_violations",
                "episode_success",
            ])

    def _check_convergence(self, episode_idx: int):
        """Mark convergence the first time the rolling mean >= 90% of best reward."""
        if self.convergence_episode is not None:
            return 
        if len(self._reward_history) < CONVERGENCE_WINDOW:
            return
        rolling_mean = np.mean(self._reward_history[-CONVERGENCE_WINDOW:])
        best = max(self._reward_history)
        if best > 0 and rolling_mean >= CONVERGENCE_THRESHOLD * best:
            self.convergence_episode = episode_idx
            self.convergence_timestep = self.num_timesteps

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_reward" in info:
                ep_reward = info["episode_reward"]
                self._reward_history.append(ep_reward)
                self._check_convergence(self.rows_written)

                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps,
                        self.rows_written,
                        ep_reward,
                        info["episode_length"],
                        info["episode_violations"],
                        info["episode_success"],
                    ])
                self.rows_written += 1
        return True