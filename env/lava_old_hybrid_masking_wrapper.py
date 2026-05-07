"""
env/lava_old_hybrid_masking_wrapper.py
=================================
old_hybrid masking wrapper for MiniGrid/LavaGap.

old_hybrid masking strategy
---------------------
* UNSAFE actions (forward into lava) are HARD-BLOCKED via the action mask —
  the safety guarantee is never relaxed.
* RISKY actions (would place agent adjacent to lava) are ALLOWED but receive
  a configurable reward PENALTY, discouraging the agent without forbidding it.

This is distinct from:
  - Hard masking  (Method 3): blocks unsafe only
  - Hybrid masking (Method 4): blocks both unsafe + risky
  - old_hybrid masking  (Method 5, this file): blocks unsafe, penalises risky

Usage with MaskablePPO
----------------------
    from env.lava_old_hybrid_masking_wrapper import make_old_hybrid_masked_env
    from sb3_contrib import MaskablePPO

    env   = make_old_hybrid_masked_env("MiniGrid-LavaGapS5-v0", seed=0)
    model = MaskablePPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=300_000)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from env.lava_logging_wrapper import LavaLoggingWrapper
from src.masking import get_action_mask, is_risky_action, get_nearby_lava_info


class Lavaold_hybridMaskingWrapper(LavaLoggingWrapper):
    """
    Extends LavaLoggingWrapper with old_hybrid masking:
      - action_masks() hard-blocks only UNSAFE actions (forward into lava).
      - Risky actions are penalised via a reward deduction, not blocked.

    Parameters
    ----------
    env          : wrapped MiniGrid environment
    risky_penalty: reward subtracted when the agent takes a risky action.
                   Default 0.1 (small relative to the +1 goal reward).
    """

    def __init__(self, env: gym.Env, risky_penalty: float = 0.1):
        super().__init__(env)
        self.risky_penalty = risky_penalty

        # Per-episode counters
        self.risky_actions_taken:  int = 0
        self.attempted_unsafe:     int = 0

        # Cumulative counters (never reset)
        self.total_risky_taken:    int = 0
        self.total_attempted_unsafe: int = 0

    # ------------------------------------------------------------------
    # Gymnasium / sb3-contrib interface
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Hard-block only UNSAFE actions (mask_risky=False).
        Risky actions remain True (allowed) — their cost comes via the
        reward penalty applied in step().
        """
        return get_action_mask(self, mask_risky=False)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.risky_actions_taken = 0
        self.attempted_unsafe    = 0
        return obs, info

    def step(self, action: int):
        from src.masking import is_unsafe_action

        # Track attempted unsafe actions (should be 0 with MaskablePPO)
        if is_unsafe_action(self, action):
            self.attempted_unsafe        += 1
            self.total_attempted_unsafe  += 1

        # Check risky BEFORE stepping (agent position will change after)
        action_is_risky = is_risky_action(self, action)
        lava_info       = get_nearby_lava_info(self)

        obs, reward, terminated, truncated, info = super().step(action)

        adjusted_reward = float(reward)

        # Apply old_hybrid penalty for risky actions
        if action_is_risky:
            adjusted_reward        -= self.risky_penalty
            self.risky_actions_taken  += 1
            self.total_risky_taken    += 1

        info = dict(info)
        info["action_is_risky"]    = action_is_risky
        info["lava_proximity"]     = lava_info
        info["n_adjacent_lava"]    = lava_info["n_adjacent_lava"]

        if terminated or truncated:
            info["risky_actions_taken"] = self.risky_actions_taken
            info["attempted_unsafe"]    = self.attempted_unsafe

        return obs, adjusted_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_old_hybrid_masked_env(
    env_id: str,
    seed: int = 0,
    risky_penalty: float = 0.1,
    render_mode: str | None = None,
) -> gym.Env:
    """
    Build the canonical training environment for old_hybrid-masked PPO experiments.

    Wrapper stack (outermost first):
        ActionMasker  →  Lavaold_hybridMaskingWrapper  →  FlatObsWrapper  →  MiniGrid

    Parameters
    ----------
    env_id        : e.g. "MiniGrid-LavaGapS5-v0"
    seed          : RNG seed
    risky_penalty : reward penalty for risky actions (default 0.1)
    render_mode   : pass "rgb_array" to record video
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
    env = Lavaold_hybridMaskingWrapper(env, risky_penalty=risky_penalty)
    env = ActionMasker(env, lambda e: e.action_masks())

    return env


def make_old_hybrid_masked_video_env(
    env_id: str,
    video_folder: str,
    seed: int = 0,
    risky_penalty: float = 0.1,
) -> gym.Env:
    """Like make_old_hybrid_masked_env but with RecordVideo for rollout recording."""
    from gymnasium.wrappers import RecordVideo
    from sb3_contrib.common.wrappers import ActionMasker
    from minigrid.wrappers import FlatObsWrapper

    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    env = FlatObsWrapper(env)
    old_hybrid_env = Lavaold_hybridMaskingWrapper(env, risky_penalty=risky_penalty)
    env = RecordVideo(old_hybrid_env, video_folder=video_folder, episode_trigger=lambda x: True)
    env = ActionMasker(env, lambda e: old_hybrid_env.action_masks())

    return env
