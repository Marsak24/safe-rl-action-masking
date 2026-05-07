"""
env/lava_soft_action_masking_wrapper.py
========================================
Gymnasium wrapper implementing **true soft action masking** for
MiniGrid / LavaGap environments.

Soft action masking strategy
-----------------------------
Unlike hard masking (which sets unsafe action probabilities to 0) or
hybrid masking (which hard-blocks unsafe actions AND penalises risky ones
via reward shaping), soft action masking:

  * Does NOT hard-block any action.  The agent CAN choose to step into lava.
  * Returns FLOAT logit adjustments from ``action_masks()``:
      -  0.0               → safe action  (no change to policy distribution)
      - -unsafe_logit_pen  → unsafe action (forward into lava; very unlikely)
      - -risky_logit_pen   → risky action  (adjacent/facing lava; somewhat unlikely)
  * These adjustments are applied additively to the policy's action logits by
    ``SoftMaskableCategorical`` before sampling — there is no reward modification.

Because no action is hard-blocked, **violations (lava entries) CAN occur
during training**.  Whether the agent learns to avoid lava is governed
entirely by the environment's natural reward signal and the logit adjustment
magnitude.

Contrast with
-------------
  - Hard masking      (env/lava_masking_wrapper.py)       : binary block → prob = 0
  - Hybrid masking    (env/lava_soft_masking_wrapper.py)  : binary block + reward penalty
  - Soft action mask  (this file)                         : float logit adjustment only

Usage
-----
    from env.lava_soft_action_masking_wrapper import make_soft_action_masked_env
    from sb3_contrib import MaskablePPO
    from agents.soft_maskable_ppo import SoftMaskableActorCriticPolicy

    env   = make_soft_action_masked_env("MiniGrid-LavaGapS5-v0", seed=0)
    model = MaskablePPO(SoftMaskableActorCriticPolicy, env, verbose=1, device="cpu")
    model.learn(total_timesteps=300_000)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from env.lava_logging_wrapper import LavaLoggingWrapper
from src.masking import N_ACTIONS, is_unsafe_action, is_risky_action


class LavaSoftActionMaskingWrapper(LavaLoggingWrapper):
    """
    Wraps a MiniGrid environment with soft action masking.

    ``action_masks()`` returns **float** logit adjustments (not binary booleans).
    No action is hard-blocked; the policy is only probabilistically discouraged
    from selecting unsafe / risky actions.

    Parameters
    ----------
    env                 : wrapped MiniGrid environment (FlatObsWrapper applied first)
    unsafe_logit_penalty: magnitude of negative logit applied to actions that would
                          move the agent directly into lava.
                          Default 5.0 → unsafe action is exp(5) ≈ 150× less likely
                          than an unpenalised action with equal policy logit.
    risky_logit_penalty : magnitude of negative logit for risky actions
                          (adjacent-to-lava, or turn to face lava).
                          Default 2.0 → exp(2) ≈ 7× less likely than safe actions.
    """

    def __init__(
        self,
        env: gym.Env,
        unsafe_logit_penalty: float = 5.0,
        risky_logit_penalty:  float = 2.0,
    ) -> None:
        super().__init__(env)
        self.unsafe_logit_penalty = unsafe_logit_penalty
        self.risky_logit_penalty  = risky_logit_penalty

    # ------------------------------------------------------------------
    # sb3-contrib / ActionMasker interface
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Return float logit adjustments for the current state.

        Shape: (N_ACTIONS,) = (7,) for MiniGrid LavaGap environments.

        Values:
          - 0.0                    → safe action (no adjustment)
          - -unsafe_logit_penalty  → action would move agent into lava
          - -risky_logit_penalty   → action places agent adjacent to / facing lava

        These are consumed by ``SoftMaskableCategorical.apply_masking`` during
        rollout collection and during the PPO update (via evaluate_actions).
        """
        adjustments = np.zeros(N_ACTIONS, dtype=np.float32)
        for action in range(N_ACTIONS):
            if is_unsafe_action(self, action):
                adjustments[action] = -self.unsafe_logit_penalty
            elif is_risky_action(self, action):
                adjustments[action] = -self.risky_logit_penalty
        return adjustments

    # ``step()`` is fully inherited from LavaLoggingWrapper.
    # Violations ARE counted there whenever the agent enters a lava cell.
    # With soft masking there is no hard block, so violations WILL occur
    # during training (especially early on) and are recorded correctly.


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_soft_action_masked_env(
    env_id: str,
    seed: int = 0,
    unsafe_logit_penalty: float = 5.0,
    risky_logit_penalty:  float = 2.0,
    render_mode: str | None = None,
) -> gym.Env:
    """
    Canonical training environment for soft action masking experiments.

    Wrapper stack (outermost → innermost):

        ActionMasker
            └─ LavaSoftActionMaskingWrapper   ← float logit adjustments
                └─ FlatObsWrapper
                    └─ MiniGrid base env

    ``ActionMasker`` is required so that MaskablePPO calls ``env.action_masks()``
    at every rollout step and stores the float adjustments in the rollout buffer.
    During PPO update, ``SoftMaskableActorCriticPolicy`` re-applies the stored
    adjustments when computing log-probs and entropy for all sampled transitions.

    Parameters
    ----------
    env_id               : e.g. "MiniGrid-LavaGapS5-v0"
    seed                 : RNG seed for the environment
    unsafe_logit_penalty : logit penalty for forward-into-lava actions
    risky_logit_penalty  : logit penalty for adjacent-to-lava / face-lava actions
    render_mode          : pass "rgb_array" when recording video
    """
    from sb3_contrib.common.wrappers import ActionMasker
    from minigrid.wrappers import FlatObsWrapper

    kwargs: dict = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env = FlatObsWrapper(env)
    env = LavaSoftActionMaskingWrapper(
        env,
        unsafe_logit_penalty=unsafe_logit_penalty,
        risky_logit_penalty=risky_logit_penalty,
    )
    env = ActionMasker(env, lambda e: e.action_masks())
    return env


def make_soft_action_masked_video_env(
    env_id: str,
    video_folder: str,
    seed: int = 0,
    unsafe_logit_penalty: float = 5.0,
    risky_logit_penalty:  float = 2.0,
) -> gym.Env:
    """Like make_soft_action_masked_env but wrapped with RecordVideo."""
    from gymnasium.wrappers import RecordVideo
    from sb3_contrib.common.wrappers import ActionMasker
    from minigrid.wrappers import FlatObsWrapper

    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    env = FlatObsWrapper(env)
    soft_env = LavaSoftActionMaskingWrapper(
        env,
        unsafe_logit_penalty=unsafe_logit_penalty,
        risky_logit_penalty=risky_logit_penalty,
    )
    env = RecordVideo(soft_env, video_folder=video_folder, episode_trigger=lambda x: True)
    env = ActionMasker(env, lambda e: soft_env.action_masks())
    return env
