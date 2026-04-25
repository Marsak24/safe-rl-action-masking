"""
tests/test_masking.py
=====================
Unit tests for src/masking.py — the hard action-masking pipeline.

Phase 1 gate
------------
The mandatory gate test is test_facing_lava_masks_forward():
    When the agent is directly facing a lava cell, the forward action
    must be disabled (mask[ACTION_FORWARD] == False).

Other tests cover:
  - Safe forward (no masking), all non-forward actions always allowed,
  - Mask never all-False (safety valve),
  - is_risky_action classification,
  - get_nearby_lava_info completeness,
  - Works across S5, S6, S7 grid sizes.

Run from the project root:
    python -m pytest tests/test_masking.py -v
"""

import sys
import os

# Ensure project root is on the path when the tests are run directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest
import gymnasium as gym
import minigrid  # noqa: F401 – registers MiniGrid environments

from minigrid.wrappers import FlatObsWrapper

from src.masking import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    N_ACTIONS,
    get_action_mask,
    get_nearby_lava_info,
    is_risky_action,
    is_unsafe_action,
)


# ---------------------------------------------------------------------------
# Helpers – craft specific environment states
# ---------------------------------------------------------------------------

def _make_env(env_id: str = "MiniGrid-LavaGapS5-v0", seed: int = 0) -> gym.Env:
    """Return a flat-obs wrapped MiniGrid env, reset with a fixed seed."""
    env = gym.make(env_id)
    env = FlatObsWrapper(env)
    env.reset(seed=seed)
    return env


def _place_agent(env: gym.Env, pos: tuple[int, int], direction: int) -> None:
    """
    Forcefully teleport the agent to *pos* facing *direction*.

    This bypasses the normal step loop so tests can assert masking
    behaviour for arbitrary hand-crafted states without replaying
    whole episodes.

    Parameters
    ----------
    pos       : (x, y) grid coordinates (not wall-adjusted)
    direction : 0=East 1=South 2=West 3=North
    """
    raw = env.unwrapped
    raw.agent_pos = np.array(pos)
    raw.agent_dir = direction


def _find_lava_cell(env: gym.Env) -> tuple[int, int] | None:
    """Return the (x, y) of the first lava cell in the grid, or None."""
    raw = env.unwrapped
    for x in range(raw.grid.width):
        for y in range(raw.grid.height):
            cell = raw.grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) == "lava":
                return (x, y)
    return None


def _place_agent_facing_lava(env: gym.Env) -> tuple[int, int]:
    """
    Move the agent to a valid non-lava cell that is directly adjacent to a
    lava cell, oriented so that the lava cell is in front.

    Returns the lava (x, y) for reference; raises RuntimeError if no such
    configuration exists in the given map.
    """
    raw    = env.unwrapped
    width  = raw.grid.width
    height = raw.grid.height

    # Direction → (dx, dy)
    DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            agent_cell = raw.grid.get(x, y)
            # Agent must stand on an empty cell (None = floor).
            if agent_cell is not None:
                continue
            for direction, (dx, dy) in DIR_VEC.items():
                fx, fy = x + dx, y + dy
                if fx < 0 or fy < 0 or fx >= width or fy >= height:
                    continue
                front_cell = raw.grid.get(fx, fy)
                if front_cell is not None and getattr(front_cell, "type", None) == "lava":
                    _place_agent(env, (x, y), direction)
                    return (fx, fy)

    raise RuntimeError("Could not find agent position adjacent to lava in this map.")


def _place_agent_facing_safe(env: gym.Env) -> tuple[int, int]:
    """
    Move the agent to a position where the cell directly ahead is empty
    (not lava, not wall, not out-of-bounds).

    Returns the agent (x, y) placed.  Raises RuntimeError if no such
    position exists (should not happen on any standard LavaGap map).
    """
    raw    = env.unwrapped
    width  = raw.grid.width
    height = raw.grid.height

    DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if raw.grid.get(x, y) is not None:
                continue  # agent cell must be floor
            for direction, (dx, dy) in DIR_VEC.items():
                fx, fy = x + dx, y + dy
                if fx < 0 or fy < 0 or fx >= width or fy >= height:
                    continue
                front_cell = raw.grid.get(fx, fy)
                if front_cell is None:  # empty floor → safe forward
                    _place_agent(env, (x, y), direction)
                    return (x, y)

    raise RuntimeError("Could not find a safe-forward position on this map.")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIsUnsafeAction:
    """Tests for is_unsafe_action()."""

    def test_facing_lava_is_unsafe(self):
        """Forward is unsafe when the agent directly faces a lava cell."""
        env = _make_env()
        _place_agent_facing_lava(env)
        assert is_unsafe_action(env, ACTION_FORWARD), (
            "is_unsafe_action must return True when agent faces lava forward."
        )

    def test_not_facing_lava_is_safe(self):
        """Forward is not unsafe when the cell ahead is not lava."""
        env = _make_env()
        _place_agent_facing_safe(env)  # finds a safe-forward cell dynamically
        assert not is_unsafe_action(env, ACTION_FORWARD), (
            "is_unsafe_action should be False when forward cell is not lava."
        )

    def test_turn_actions_never_unsafe(self):
        """Turning left/right is never classified as unsafe."""
        env = _make_env()
        _place_agent_facing_lava(env)  # worst case: facing lava
        assert not is_unsafe_action(env, ACTION_LEFT)
        assert not is_unsafe_action(env, ACTION_RIGHT)

    def test_non_move_actions_never_unsafe(self):
        """Pickup, drop, toggle, done are never unsafe."""
        env = _make_env()
        _place_agent_facing_lava(env)
        for action in (3, 4, 5, 6):
            assert not is_unsafe_action(env, action), (
                f"Action {action} should never be classified as unsafe."
            )


class TestGetActionMask:
    """Tests for get_action_mask() — the Phase 1 gate and related checks."""

    def test_phase1_gate_forward_masked_when_facing_lava(self):
        """
        PHASE 1 GATE TEST
        -----------------
        When the agent faces a lava cell, the forward action must be False
        (disabled) in the mask returned by get_action_mask().
        """
        env = _make_env()
        _place_agent_facing_lava(env)

        mask = get_action_mask(env)

        assert mask.dtype == bool, "Mask must be a boolean np.ndarray."
        assert len(mask) == N_ACTIONS, f"Mask length must be {N_ACTIONS}."
        assert not mask[ACTION_FORWARD], (
            "PHASE 1 GATE FAILED: mask[ACTION_FORWARD] must be False when "
            "the agent is facing lava."
        )

    def test_forward_allowed_when_safe(self):
        """When the cell ahead is not lava, forward must remain allowed."""
        env = _make_env()
        _place_agent_facing_safe(env)  # finds a safe-forward cell dynamically
        mask = get_action_mask(env)
        assert mask[ACTION_FORWARD], (
            "Forward should be True (allowed) when there is no lava ahead."
        )

    def test_turn_actions_always_allowed_in_hard_mode(self):
        """In hard masking (mask_risky=False), turns are never blocked."""
        env = _make_env()
        _place_agent_facing_lava(env)
        mask = get_action_mask(env, mask_risky=False)
        assert mask[ACTION_LEFT],  "Turn left must remain allowed in hard masking."
        assert mask[ACTION_RIGHT], "Turn right must remain allowed in hard masking."

    def test_mask_never_all_false(self):
        """The mask must never be entirely False (safety valve)."""
        env = _make_env()
        # Even in worst case (all actions blocked hypothetically), at least
        # one action is available.
        mask = get_action_mask(env)
        assert mask.any(), (
            "get_action_mask must always return at least one True entry."
        )

    def test_mask_shape(self):
        """Mask shape equals the action space size."""
        for env_id in (
            "MiniGrid-LavaGapS5-v0",
            "MiniGrid-LavaGapS6-v0",
            "MiniGrid-LavaGapS7-v0",
        ):
            env  = _make_env(env_id)
            mask = get_action_mask(env)
            expected = env.unwrapped.action_space.n
            assert len(mask) == expected, (
                f"{env_id}: mask length {len(mask)} != action space {expected}"
            )
            env.close()

    def test_mask_risky_flag_blocks_facing_lava_after_turn(self):
        """
        With mask_risky=True, turning to face lava should be masked out
        when a turn would cause the agent to face lava.
        """
        env = _make_env()
        raw = env.unwrapped
        width, height = raw.grid.width, raw.grid.height

        DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

        found = False
        # Find a position where turning LEFT would make agent face lava.
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if raw.grid.get(x, y) is not None:
                    continue
                for cur_dir in range(4):
                    left_dir = (cur_dir - 1) % 4
                    ldx, ldy = DIR_VEC[left_dir]
                    left_front_cell = raw.grid.get(x + ldx, y + ldy)
                    if (
                        left_front_cell is not None
                        and getattr(left_front_cell, "type", None) == "lava"
                    ):
                        # Also ensure we are NOT already facing lava forward.
                        fdx, fdy = DIR_VEC[cur_dir]
                        front_cell = raw.grid.get(x + fdx, y + fdy)
                        if (
                            front_cell is None
                            or getattr(front_cell, "type", None) != "lava"
                        ):
                            _place_agent(env, (x, y), cur_dir)
                            mask_soft = get_action_mask(env, mask_risky=True)
                            assert not mask_soft[ACTION_LEFT], (
                                "mask_risky=True must block turning left when "
                                "the resulting front cell is lava."
                            )
                            found = True
                            break
                if found:
                    break
            if found:
                break

        if not found:
            pytest.skip(
                "No position found where turning left would face lava on this map."
            )


class TestIsRiskyAction:
    """Tests for is_risky_action()."""

    def test_risky_forward_to_lava_neighbour(self):
        """
        Moving forward to a cell adjacent to lava (but not into lava)
        should be classified as risky.
        """
        env = _make_env()
        raw = env.unwrapped
        width, height = raw.grid.width, raw.grid.height

        DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

        found = False
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if raw.grid.get(x, y) is not None:
                    continue
                for cur_dir in range(4):
                    fdx, fdy = DIR_VEC[cur_dir]
                    fx, fy = x + fdx, y + fdy
                    if fx < 0 or fy < 0 or fx >= width or fy >= height:
                        continue
                    front_cell = raw.grid.get(fx, fy)
                    # Forward must go to an empty cell (not lava, not wall).
                    if front_cell is not None:
                        continue
                    # Check that (fx, fy) has a lava neighbour.
                    for d2 in range(4):
                        dx2, dy2 = DIR_VEC[d2]
                        nx, ny = fx + dx2, fy + dy2
                        nc = raw.grid.get(nx, ny)
                        if nc is not None and getattr(nc, "type", None) == "lava":
                            _place_agent(env, (x, y), cur_dir)
                            assert is_risky_action(env, ACTION_FORWARD), (
                                "Moving forward to a lava-adjacent cell should be risky."
                            )
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break

        if not found:
            pytest.skip("No lava-adjacent empty cell found on this map.")

    def test_risky_does_not_flag_safe_forward(self):
        """
        Forward to a cell with no lava neighbours should not be risky.
        We search for such a position dynamically rather than hardcoding
        a coordinate that may vary by map seed or size.
        """
        env = _make_env()
        raw = env.unwrapped
        width, height = raw.grid.width, raw.grid.height
        DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

        found = False
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if raw.grid.get(x, y) is not None:
                    continue
                for cur_dir, (dx, dy) in DIR_VEC.items():
                    fx, fy = x + dx, y + dy
                    if fx < 0 or fy < 0 or fx >= width or fy >= height:
                        continue
                    if raw.grid.get(fx, fy) is not None:  # must be empty floor
                        continue
                    # Check (fx, fy) has NO lava neighbours.
                    neighbours_lava = any(
                        raw.grid.get(fx + ddx, fy + ddy) is not None
                        and getattr(raw.grid.get(fx + ddx, fy + ddy), "type", None) == "lava"
                        for ddx, ddy in DIR_VEC.values()
                    )
                    if not neighbours_lava:
                        _place_agent(env, (x, y), cur_dir)
                        assert not is_risky_action(env, ACTION_FORWARD), (
                            "Forward to a lava-free neighbourhood must not be risky."
                        )
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if not found:
            pytest.skip("No lava-isolated forward target found on this map.")


class TestGetNearbyLavaInfo:
    """Tests for get_nearby_lava_info()."""

    def test_returns_all_keys(self):
        env = _make_env()
        info = get_nearby_lava_info(env)
        expected_keys = {
            "front_is_lava",
            "left_is_lava",
            "right_is_lava",
            "back_is_lava",
            "n_adjacent_lava",
            "would_face_lava_left",
            "would_face_lava_right",
        }
        assert expected_keys == set(info.keys()), (
            f"Missing keys: {expected_keys - set(info.keys())}"
        )

    def test_front_is_lava_consistent_with_mask(self):
        """front_is_lava must match the mask's forward entry."""
        env = _make_env()
        _place_agent_facing_lava(env)
        info = get_nearby_lava_info(env)
        mask = get_action_mask(env)
        assert info["front_is_lava"] == (not mask[ACTION_FORWARD]), (
            "front_is_lava and mask[FORWARD] must be consistent."
        )

    def test_n_adjacent_lava_matches_sum(self):
        """n_adjacent_lava must equal the sum of the four directional flags."""
        env = _make_env()
        _place_agent_facing_lava(env)
        info = get_nearby_lava_info(env)
        total = sum([
            info["front_is_lava"],
            info["left_is_lava"],
            info["right_is_lava"],
            info["back_is_lava"],
        ])
        assert info["n_adjacent_lava"] == total


class TestGridSizeIndependence:
    """Verify masking works across LavaGapS5, S6, S7."""

    @pytest.mark.parametrize("env_id", [
        "MiniGrid-LavaGapS5-v0",
        "MiniGrid-LavaGapS6-v0",
        "MiniGrid-LavaGapS7-v0",
    ])
    def test_forward_masked_facing_lava_all_sizes(self, env_id):
        """Phase 1 gate: forward masked when facing lava on all three maps."""
        env = _make_env(env_id)
        _place_agent_facing_lava(env)
        mask = get_action_mask(env)
        assert not mask[ACTION_FORWARD], (
            f"{env_id}: forward must be masked when agent faces lava."
        )
        env.close()

    @pytest.mark.parametrize("env_id", [
        "MiniGrid-LavaGapS5-v0",
        "MiniGrid-LavaGapS6-v0",
        "MiniGrid-LavaGapS7-v0",
    ])
    def test_mask_never_all_false_all_sizes(self, env_id):
        env = _make_env(env_id)
        mask = get_action_mask(env)
        assert mask.any(), f"{env_id}: mask must never be all-False."
        env.close()
