import gymnasium as gym
import numpy as np


class LavaLoggingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_violations = 0
        self.episode_success = 0
        self._visit_counts = None
        self._init_visit_map()

    def _init_visit_map(self):
        grid = self.unwrapped.grid
        if grid is not None:
            self._visit_counts = np.zeros((grid.width, grid.height), dtype=np.int32)

    @property
    def visit_counts(self) -> np.ndarray:

        return self._visit_counts

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_violations = 0
        self.episode_success = 0
        if self._visit_counts is None:
            self._init_visit_map()
        self._log_visit()
        return obs, info

    def _log_visit(self):
        if self._visit_counts is None:
            return
        x, y = self.unwrapped.agent_pos
        self._visit_counts[x, y] += 1

    def _on_lava(self):
        x, y = self.unwrapped.agent_pos
        cell = self.unwrapped.grid.get(x, y)
        return cell is not None and getattr(cell, "type", None) == "lava"

    def _on_goal(self):
        x, y = self.unwrapped.agent_pos
        cell = self.unwrapped.grid.get(x, y)
        return cell is not None and getattr(cell, "type", None) == "goal"

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += float(reward)
        self.episode_length += 1
        self._log_visit()

        if self._on_lava():
            self.episode_violations += 1
        if self._on_goal():
            self.episode_success = 1

        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = self.episode_reward
            info["episode_length"] = self.episode_length
            info["episode_violations"] = self.episode_violations
            info["episode_success"] = self.episode_success

        return obs, reward, terminated, truncated, info