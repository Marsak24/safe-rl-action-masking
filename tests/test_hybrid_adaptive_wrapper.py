"""
tests/test_hybrid_wrapper.py
============================
Focused tests for hybrid variant 1.

The hybrid wrapper should:
  * block forward into lava through action_masks()
  * keep risky actions available
  * subtract risky_penalty when a risky action is actually taken
"""

import os
import sys

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments
import numpy as np
import pytest
from minigrid.wrappers import FlatObsWrapper

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.lava_hybrid_adaptive_wrapper import LavaHybridWrapper
from src.masking import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, is_risky_action


DIR_VEC = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


def _make_env(risky_penalty: float = 0.25, **kwargs):
    env = gym.make("MiniGrid-LavaGapS5-v0")
    env = FlatObsWrapper(env)
    env.reset(seed=0)
    return LavaHybridWrapper(env, risky_penalty=risky_penalty, **kwargs)


def _place_agent(env, pos: tuple[int, int], direction: int) -> None:
    raw = env.unwrapped
    raw.agent_pos = np.array(pos)
    raw.agent_dir = direction


def _place_agent_facing_lava(env) -> None:
    raw = env.unwrapped
    for x in range(1, raw.grid.width - 1):
        for y in range(1, raw.grid.height - 1):
            if raw.grid.get(x, y) is not None:
                continue
            for direction, (dx, dy) in DIR_VEC.items():
                cell = raw.grid.get(x + dx, y + dy)
                if cell is not None and getattr(cell, "type", None) == "lava":
                    _place_agent(env, (x, y), direction)
                    return
    raise RuntimeError("Could not find a position facing lava.")


def _place_agent_with_risky_turn(env) -> int:
    raw = env.unwrapped
    for x in range(1, raw.grid.width - 1):
        for y in range(1, raw.grid.height - 1):
            if raw.grid.get(x, y) is not None:
                continue
            for cur_dir in range(4):
                for action in (ACTION_LEFT, ACTION_RIGHT):
                    next_dir = (cur_dir - 1) % 4 if action == ACTION_LEFT else (cur_dir + 1) % 4
                    dx, dy = DIR_VEC[next_dir]
                    cell = raw.grid.get(x + dx, y + dy)
                    if cell is not None and getattr(cell, "type", None) == "lava":
                        _place_agent(env, (x, y), cur_dir)
                        if is_risky_action(env, action):
                            return action
    raise RuntimeError("Could not find a risky turn position.")


def test_hybrid_mask_blocks_unsafe_forward():
    env = _make_env()
    _place_agent_facing_lava(env)

    mask = env.action_masks()

    assert not mask[ACTION_FORWARD]
    env.close()


def test_hybrid_keeps_risky_turn_available_but_penalizes_it():
    penalty = 0.25
    env = _make_env(risky_penalty=penalty)
    action = _place_agent_with_risky_turn(env)

    mask = env.action_masks()
    assert mask[action], "Hybrid variant 1 should not mask risky actions."

    _, reward, _, _, info = env.step(action)

    assert info["action_was_risky"]
    assert info["risky_penalty_applied"] == pytest.approx(penalty)
    assert env.risky_actions == 1
    assert env.risky_penalty_total == pytest.approx(penalty)
    assert reward <= -penalty
    env.close()


def test_adaptive_hybrid_penalty_increases_with_training_steps():
    env = _make_env(
        risky_penalty_start=0.01,
        risky_penalty_end=0.10,
        risky_penalty_schedule_steps=10,
    )

    assert env.current_risky_penalty == pytest.approx(0.01)

    env.training_steps = 5
    assert env.current_risky_penalty == pytest.approx(0.055)
    assert env.penalty_schedule_progress == pytest.approx(0.5)

    env.training_steps = 20
    assert env.current_risky_penalty == pytest.approx(0.10)
    assert env.penalty_schedule_progress == pytest.approx(1.0)
    env.close()


def test_adaptive_hybrid_applies_current_penalty_to_risky_action():
    env = _make_env(
        risky_penalty_start=0.01,
        risky_penalty_end=0.10,
        risky_penalty_schedule_steps=10,
    )
    env.training_steps = 5
    action = _place_agent_with_risky_turn(env)

    _, _, _, _, info = env.step(action)

    assert info["action_was_risky"]
    assert info["current_risky_penalty"] == pytest.approx(0.055)
    assert info["risky_penalty_applied"] == pytest.approx(0.055)
    env.close()
