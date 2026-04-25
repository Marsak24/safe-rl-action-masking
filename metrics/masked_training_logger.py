"""
metrics/masked_training_logger.py
==================================
Extended EpisodeCSVLogger that additionally records per-episode masking
counters (masked_unsafe_attempts, masked_risky_attempts).

Designed to be a drop-in replacement for EpisodeCSVLogger when training
with LavaMaskingWrapper + MaskablePPO.
"""

from __future__ import annotations

import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

CONVERGENCE_WINDOW           = 20
CONVERGENCE_REWARD_THRESHOLD = 0.7


class MaskedEpisodeCSVLogger(BaseCallback):
    """
    Callback that writes one CSV row per completed episode.

    Columns
    -------
    timestep, episode, episode_reward, episode_length, episode_violations,
    episode_success, masked_unsafe_attempts, masked_risky_attempts

    The masking columns come from the info dict populated by
    LavaMaskingWrapper.  If they are absent (e.g. when used with a
    non-masking env) they default to 0.
    """

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path    = csv_path
        self.rows_written = 0

        self._reward_history: list[float]  = []
        self.convergence_episode: int | None  = None
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
                "masked_risky_attempts",
            ])

    def _check_convergence(self, episode_idx: int) -> None:
        if self.convergence_episode is not None:
            return
        if len(self._reward_history) < CONVERGENCE_WINDOW:
            return
        rolling_mean = np.mean(self._reward_history[-CONVERGENCE_WINDOW:])
        if rolling_mean >= CONVERGENCE_REWARD_THRESHOLD:
            self.convergence_episode  = episode_idx
            self.convergence_timestep = self.num_timesteps

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_reward" in info:
                ep_reward = info["episode_reward"]
                self._reward_history.append(ep_reward)
                self._check_convergence(self.rows_written)

                masked_unsafe = info.get("masked_unsafe_attempts", 0)
                masked_risky  = info.get("masked_risky_attempts",  0)

                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps,
                        self.rows_written,
                        ep_reward,
                        info["episode_length"],
                        info["episode_violations"],
                        info["episode_success"],
                        masked_unsafe,
                        masked_risky,
                    ])
                self.rows_written += 1
        return True
