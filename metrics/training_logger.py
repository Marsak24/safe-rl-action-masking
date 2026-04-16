import os
import csv
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeCSVLogger(BaseCallback):
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.rows_written = 0

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep",
                "episode_reward",
                "episode_length",
                "episode_violations",
                "episode_success",
            ])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_reward" in info:
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps,
                        info["episode_reward"],
                        info["episode_length"],
                        info["episode_violations"],
                        info["episode_success"],
                    ])
                self.rows_written += 1
        return True
