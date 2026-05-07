"""
metrics/hybrid_training_logger.py
=================================
Episode CSV logger for hybrid PPO.

It keeps the standard training metrics used by the other methods and adds
hybrid-specific columns for hard-mask interventions and risky-action penalties.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

CONVERGENCE_WINDOW = 20
CONVERGENCE_REWARD_THRESHOLD = 0.7


class HybridEpisodeCSVLogger(BaseCallback):
    """
    Write one CSV row per completed episode.

    Columns
    -------
    timestep, episode, episode_reward, episode_length, episode_violations,
    episode_success, masked_unsafe_attempts, risky_actions,
    risky_penalty_total
    """

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.rows_written = 0

        self._reward_history: list[float] = []
        self.convergence_episode: int | None = None
        self.convergence_timestep: int | None = None

        d = os.path.dirname(csv_path)
        if d:
            os.makedirs(d, exist_ok=True)

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep",
                "episode",
                "episode_reward",
                "episode_length",
                "episode_violations",
                "episode_success",
                "masked_unsafe_attempts",
                "risky_actions",
                "risky_penalty_total",
            ])

    def _check_convergence(self, episode_idx: int) -> None:
        if self.convergence_episode is not None:
            return
        if len(self._reward_history) < CONVERGENCE_WINDOW:
            return

        rolling_mean = np.mean(self._reward_history[-CONVERGENCE_WINDOW:])
        if rolling_mean >= CONVERGENCE_REWARD_THRESHOLD:
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
                        info.get("masked_unsafe_attempts", 0),
                        info.get("risky_actions", 0),
                        info.get("risky_penalty_total", 0.0),
                    ])
                self.rows_written += 1
        return True
