"""
env/lava_masking_wrapper.py
===========================
Gymnasium wrapper that integrates hard action masking with MiniGrid/LavaGap.

This wrapper:
  1. Exposes action_masks() so MaskablePPO (sb3-contrib) can call it at
     every rollout step without any extra plumbing.
  2. Inherits all logging from LavaLoggingWrapper (episode reward, length,
     violations, visit counts).
  3. Adds per-episode counters for *masked unsafe attempts* and
     *masked risky attempts* so the team can analyse how often the mask
     fires and compare safety behaviour across methods.
  4. Supports both hard masking (Method 3, mask_risky=False) and hybrid
     masking (Method 4, mask_risky=True) via a single constructor flag.

Wrapper stack (outside → inside)
---------------------------------
    ActionMasker (sb3-contrib, optional – see make_masked_env())
        LavaMaskingWrapper          ← this file
            FlatObsWrapper
                MiniGrid base env

Usage
-----
    from env.lava_masking_wrapper import LavaMaskingWrapper, make_masked_env
    from sb3_contrib import MaskablePPO

    env   = make_masked_env("MiniGrid-LavaGapS5-v0", seed=0)
    model = MaskablePPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=300_000)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from env.lava_logging_wrapper import LavaLoggingWrapper
from src.masking import get_action_mask, get_nearby_lava_info


class LavaMaskingWrapper(LavaLoggingWrapper):
    """
    Extends LavaLoggingWrapper with action-masking capabilities.

    Parameters
    ----------
    env        : the wrapped environment (must have MiniGrid as unwrapped base)
    mask_risky : if True, also mask *risky* actions (Method 4 hybrid mode).
                 Default False = hard masking only (Method 3).
    """

    def __init__(self, env: gym.Env, mask_risky: bool = False):
        super().__init__(env)
        self.mask_risky = mask_risky

        # Per-episode counters
        self.masked_unsafe_attempts: int = 0
        self.masked_risky_attempts:  int = 0

        # Cumulative counters across all episodes (never reset)
        self.total_masked_unsafe: int = 0
        self.total_masked_risky:  int = 0

    # ------------------------------------------------------------------
    # Gymnasium / sb3-contrib interface
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Return the boolean action mask for the current state.

        Called automatically by MaskablePPO at every rollout step.
        True  = allowed, False = blocked.

        Delegates to src.masking.get_action_mask so that any downstream
        consumer (wrappers, callbacks, tests) always uses the same logic.
        """
        return get_action_mask(self, mask_risky=self.mask_risky)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.masked_unsafe_attempts = 0
        self.masked_risky_attempts  = 0
        return obs, info

    def step(self, action: int):
        # -----------------------------------------------------------------
        # Log attempted unsafe / risky actions BEFORE the step so we can
        # record what the agent *tried* to do (not what the mask allowed).
        # -----------------------------------------------------------------
        from src.masking import is_unsafe_action, is_risky_action

        if is_unsafe_action(self, action):
            self.masked_unsafe_attempts += 1
            self.total_masked_unsafe    += 1

        if self.mask_risky and is_risky_action(self, action):
            self.masked_risky_attempts += 1
            self.total_masked_risky    += 1

        # -----------------------------------------------------------------
        # Capture pre-step lava proximity for per-step info enrichment
        # -----------------------------------------------------------------
        lava_info = get_nearby_lava_info(self)

        obs, reward, terminated, truncated, info = super().step(action)

        # -----------------------------------------------------------------
        # Attach masking diagnostics to the info dict
        # -----------------------------------------------------------------
        info = dict(info)
        info["lava_proximity"]         = lava_info
        info["action_was_unsafe"]      = bool(is_unsafe_action(self, action))
        info["n_adjacent_lava"]        = lava_info["n_adjacent_lava"]

        if terminated or truncated:
            info["masked_unsafe_attempts"] = self.masked_unsafe_attempts
            info["masked_risky_attempts"]  = self.masked_risky_attempts

        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_masked_env(
    env_id: str,
    seed: int = 0,
    mask_risky: bool = False,
    render_mode: str | None = None,
) -> gym.Env:
    """
    Build the canonical training environment for masked-PPO experiments.

    Wrapper stack (outermost first):
        ActionMasker  →  LavaMaskingWrapper  →  FlatObsWrapper  →  MiniGrid

    Parameters
    ----------
    env_id      : e.g. "MiniGrid-LavaGapS5-v0"
    seed        : RNG seed applied to both env.reset and action_space
    mask_risky  : passed through to LavaMaskingWrapper (Method 4 flag)
    render_mode : pass "rgb_array" to record video

    Returns
    -------
    gymnasium.Env ready for use with MaskablePPO
    """
    from sb3_contrib.common.wrappers import ActionMasker
    from minigrid.wrappers import FlatObsWrapper

    kwargs = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env = FlatObsWrapper(env)
    env = LavaMaskingWrapper(env, mask_risky=mask_risky)

    # ActionMasker tells MaskablePPO where to find the mask at each step.
    env = ActionMasker(env, lambda e: e.action_masks())

    return env


def make_masked_video_env(
    env_id: str,
    video_folder: str,
    seed: int = 0,
    mask_risky: bool = False,
) -> gym.Env:
    """Like make_masked_env but with RecordVideo for rollout recording."""
    from gymnasium.wrappers import RecordVideo
    from sb3_contrib.common.wrappers import ActionMasker
    from minigrid.wrappers import FlatObsWrapper

    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    env = FlatObsWrapper(env)
    masking_env = LavaMaskingWrapper(env, mask_risky=mask_risky)
    env = RecordVideo(masking_env, video_folder=video_folder, episode_trigger=lambda x: True)
    # ActionMasker must be outermost; the lambda reaches through RecordVideo
    # to the LavaMaskingWrapper which owns action_masks().
    env = ActionMasker(env, lambda e: masking_env.action_masks())

    return env
