"""
agents/soft_maskable_ppo.py
============================
Custom PPO components for **true soft action masking**.

True soft action masking (distinct from hybrid masking) adjusts action
selection PROBABILISTICALLY at the policy/distribution level:

  - Unsafe actions (forward into lava):  severe negative logit adjustment
  - Risky actions  (adjacent-to-lava):   mild negative logit adjustment
  - Safe actions:                         no adjustment (0.0)

The agent CAN still select unsafe or risky actions — they are never
hard-blocked — but their selection probability is reduced by exp(penalty).
This allows the policy gradient to experience and back-propagate through
lava-entry events, unlike hard masking which zeros that gradient entirely.

Contrast with other implemented methods
----------------------------------------
  - Hard masking  (Method 3): binary mask → probability = 0 (no gradient)
  - Hybrid masking (Method 4/5): binary mask + REWARD penalty (gradient
      flows via the reward signal, but not through the action distribution)
  - Soft action masking (this module): logit adjustment only — no hard
      block, no reward shaping; safety signal lives in the distribution

Components
----------
SoftMaskableCategorical              -- distribution applying float logit adjustments
SoftMaskableCategoricalDistribution  -- distribution factory / wrapper
SoftMaskableActorCriticPolicy        -- policy that uses the float distribution

Usage
-----
    from sb3_contrib import MaskablePPO
    from agents.soft_maskable_ppo import SoftMaskableActorCriticPolicy

    model = MaskablePPO(
        policy = SoftMaskableActorCriticPolicy,
        env    = env,   # must be wrapped with ActionMasker returning float adjustments
        **PPO_KWARGS,
    )
"""

from __future__ import annotations

import numpy as np
import torch as th
from torch.distributions import Categorical

from sb3_contrib.common.maskable.distributions import (
    MaskableCategorical,
    MaskableCategoricalDistribution,
    MaybeMasks,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


# ---------------------------------------------------------------------------
# Distribution layer
# ---------------------------------------------------------------------------

class SoftMaskableCategorical(MaskableCategorical):
    """
    Categorical distribution that consumes **float** logit adjustments instead
    of the binary masks expected by MaskableCategorical.

    ``apply_masking`` treats the incoming array as additive logit offsets:

        adjusted_logit[i] = original_logit[i] + adjustment[i]

    Typical adjustment values
    -------------------------
    - 0.0                → safe action (no change to probability)
    - -2.0               → risky action (~7× less likely; can still be chosen)
    - -5.0               → unsafe action (~150× less likely; can still be chosen)

    No action probability is forced to exactly 0, so the policy gradient
    flows through all actions including unsafe ones during training.
    """

    def apply_masking(self, masks: MaybeMasks) -> None:
        """
        Apply floating-point logit adjustments in-place.

        :param masks: Float array/tensor shaped compatibly with
                      ``self._original_logits``.  0.0 = no change; negative
                      values reduce the action's log-probability.
                      Pass ``None`` to restore the unmodified distribution.
        """
        if masks is not None:
            device = self._original_logits.device
            adj = th.as_tensor(
                masks, dtype=self._original_logits.dtype, device=device
            ).reshape(self._original_logits.shape)
            logits = self._original_logits + adj
        else:
            logits = self._original_logits

        # Clear ``self.masks`` so that entropy() uses the standard Categorical
        # formula — all actions have non-zero probability, so the special
        # masked-entropy calculation in MaskableCategorical is not needed.
        self.masks = None

        # Reinitialise the Categorical base with updated logits.
        # We bypass MaskableCategorical.__init__ so that _original_logits is
        # not overwritten by the reinitialisation.
        self.__dict__.pop("probs", None)
        Categorical.__init__(self, logits=logits, validate_args=self._validate_args)


class SoftMaskableCategoricalDistribution(MaskableCategoricalDistribution):
    """
    Distribution factory that builds ``SoftMaskableCategorical`` instances
    instead of ``MaskableCategorical``.

    Drop-in replacement for ``MaskableCategoricalDistribution`` in any
    MaskableActorCriticPolicy subclass.
    """

    def proba_distribution(
        self, action_logits: th.Tensor
    ) -> "SoftMaskableCategoricalDistribution":
        self.distribution = SoftMaskableCategorical(logits=action_logits)
        return self


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class SoftMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """
    MaskableActorCriticPolicy variant that swaps the hard-binary categorical
    distribution for ``SoftMaskableCategoricalDistribution`` after the
    parent builds the network.

    All other policy internals — MLP extractor, action_net, value_net,
    and the optimizer — are unchanged and built identically to the standard
    MaskableActorCriticPolicy.

    When passed to MaskablePPO, the policy will:
      1. Query the env's ``action_masks()`` to get float logit adjustments.
      2. Add the adjustments to the action logits before softmax / sampling.
      3. Reapply the stored adjustments during the PPO update step
         (``evaluate_actions``) so that log-probs and entropy are computed
         consistently with the adjusted distribution.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # action_net (nn.Linear → n_actions outputs) was built correctly by
        # super().__init__; we only need to replace the distribution factory.
        self.action_dist = SoftMaskableCategoricalDistribution(
            int(self.action_space.n)
        )
