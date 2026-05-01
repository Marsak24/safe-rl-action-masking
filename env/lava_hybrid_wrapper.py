"""
env/lava_hybrid_wrapper.py
==========================
Hybrid safety wrapper for MiniGrid/LavaGap experiments.

Hybrid variant 1:
  * Hard-mask unsafe actions: moving forward directly into lava is blocked.
  * Penalize risky actions: actions that move near lava or make the agent face
    lava remain available, but receive an extra negative reward.

This gives the agent a hard safety boundary for immediate lava entry while
still using reward shaping to discourage dangerous positions/orientations.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from env.lava_masking_wrapper import LavaMaskingWrapper
from src.masking import (
    get_action_mask,
    get_nearby_lava_info,
    is_risky_action,
    is_unsafe_action,
)


class LavaHybridWrapper(LavaMaskingWrapper):
    """
    Hard action masking for unsafe actions plus reward penalty for risky ones.

    Parameters
    ----------
    env : gym.Env
        Wrapped MiniGrid environment. Usually FlatObsWrapper(MiniGrid).
    risky_penalty : float
        Amount subtracted from reward whenever the chosen action is risky.
    """

    def __init__(self, env: gym.Env, risky_penalty: float = 0.1):
        super().__init__(env, mask_risky=False)
        self.risky_penalty = risky_penalty

        self.masked_unsafe_attempts: int = 0
        self.risky_actions: int = 0
        self.risky_penalty_total: float = 0.0

        self.total_masked_unsafe: int = 0
        self.total_risky_actions: int = 0
        self.total_risky_penalty: float = 0.0

    def action_masks(self) -> np.ndarray:
        """
        Return hard safety mask.

        True means allowed, False means blocked. For hybrid variant 1, only
        unsafe actions are blocked; risky actions remain available and are
        handled through reward shaping in step().
        """
        return get_action_mask(self, mask_risky=False)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.masked_unsafe_attempts = 0
        self.risky_actions = 0
        self.risky_penalty_total = 0.0
        return obs, info

    def step(self, action: int):
        action_is_unsafe = is_unsafe_action(self, action)
        action_is_risky = is_risky_action(self, action)
        lava_info = get_nearby_lava_info(self)

        if action_is_unsafe:
            self.masked_unsafe_attempts += 1
            self.total_masked_unsafe += 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        adjusted_reward = float(reward)
        risky_penalty_applied = 0.0

        if action_is_risky:
            risky_penalty_applied = float(self.risky_penalty)
            adjusted_reward -= risky_penalty_applied
            self.risky_actions += 1
            self.total_risky_actions += 1
            self.risky_penalty_total += risky_penalty_applied
            self.total_risky_penalty += risky_penalty_applied

        self.episode_reward += adjusted_reward
        self.episode_length += 1
        self._log_visit()

        if self._on_lava():
            self.episode_violations += 1
        if self._on_goal():
            self.episode_success = 1

        info = dict(info)
        info["lava_proximity"] = lava_info
        info["action_was_unsafe"] = bool(action_is_unsafe)
        info["action_was_risky"] = bool(action_is_risky)
        info["n_adjacent_lava"] = lava_info["n_adjacent_lava"]
        info["risky_penalty_applied"] = risky_penalty_applied

        if terminated or truncated:
            info["episode_reward"] = self.episode_reward
            info["episode_length"] = self.episode_length
            info["episode_violations"] = self.episode_violations
            info["episode_success"] = self.episode_success
            info["masked_unsafe_attempts"] = self.masked_unsafe_attempts
            info["risky_actions"] = self.risky_actions
            info["risky_penalty_total"] = self.risky_penalty_total

        return obs, adjusted_reward, terminated, truncated, info


def make_hybrid_env(
    env_id: str,
    seed: int = 0,
    risky_penalty: float = 0.1,
    render_mode: str | None = None,
) -> gym.Env:
    """
    Build the canonical environment for hybrid-PPO experiments.

    Wrapper stack:
        ActionMasker -> LavaHybridWrapper -> FlatObsWrapper -> MiniGrid
    """
    from minigrid.wrappers import FlatObsWrapper
    from sb3_contrib.common.wrappers import ActionMasker

    kwargs = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env = FlatObsWrapper(env)
    env = LavaHybridWrapper(env, risky_penalty=risky_penalty)
    env = ActionMasker(env, lambda e: e.action_masks())

    return env


def make_hybrid_video_env(
    env_id: str,
    video_folder: str,
    seed: int = 0,
    risky_penalty: float = 0.1,
) -> gym.Env:
    """Like make_hybrid_env but with RecordVideo for rollout recording."""
    from gymnasium.wrappers import RecordVideo
    from minigrid.wrappers import FlatObsWrapper
    from sb3_contrib.common.wrappers import ActionMasker

    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env = FlatObsWrapper(env)
    env = LavaHybridWrapper(env, risky_penalty=risky_penalty)
    env = ActionMasker(env, lambda e: e.action_masks())
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)

    return env
