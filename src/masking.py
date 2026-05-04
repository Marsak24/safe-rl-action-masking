"""
src/masking.py
==============
Core action-masking logic for safe RL on MiniGrid/LavaGap environments.

PUBLIC API
----------
get_action_mask(env, mask_risky=False) -> np.ndarray[bool, (n_actions,)]
    Returns True for every action that is ALLOWED, False for every action
    that must be blocked.  Call this from any wrapper or callback.

is_unsafe_action(env, action) -> bool
    Project-contract definition: True when the action would move the agent
    directly into a lava cell (always blocked by hard masking).

is_risky_action(env, action) -> bool
    True when the action would place the agent in a cell that neighbours
    lava (or would cause the agent to face lava after a turn).
    Used by Method 4 (hybrid) and Method 5 (old_hybrid masking); not blocked by
    hard masking in Method 3.

get_nearby_lava_info(env) -> dict
    Returns a structured summary of lava proximity.  Intended for logging
    and hybrid-masking consumers.

DESIGN NOTES
------------
* All functions accept any gym.Env wrapper chain.  They call env.unwrapped
  internally so they always reach the raw MiniGrid environment regardless
  of how many wrappers are stacked on top.
* No hard-coded grid size: works for LavaGapS5, S6, S7, and any other
  MiniGrid variant that exposes agent_pos / agent_dir / grid.
* The function never returns an all-False mask: if every computed mask
  entry would be False, it falls back to allowing all actions so training
  cannot deadlock.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

# ---------------------------------------------------------------------------
# MiniGrid action indices (mirror minigrid.core.actions.Actions)
# ---------------------------------------------------------------------------
ACTION_LEFT    = 0   # rotate agent left (counter-clockwise)
ACTION_RIGHT   = 1   # rotate agent right (clockwise)
ACTION_FORWARD = 2   # move one step forward
ACTION_PICKUP  = 3
ACTION_DROP    = 4
ACTION_TOGGLE  = 5
ACTION_DONE    = 6

N_ACTIONS = 7  # total size of the discrete action space

# ---------------------------------------------------------------------------
# Direction vectors: agent_dir (0-3) → (dx, dy) in grid coordinates
# MiniGrid convention: x grows right, y grows down
# dir=0 East, dir=1 South, dir=2 West, dir=3 North
# ---------------------------------------------------------------------------
_DIR_TO_VEC: dict[int, np.ndarray] = {
    0: np.array([1,  0]),   # East  (right)
    1: np.array([0,  1]),   # South (down)
    2: np.array([-1, 0]),   # West  (left)
    3: np.array([0, -1]),   # North (up)
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _unwrapped(env: gym.Env):
    """Return the raw MiniGrid environment regardless of wrapper depth."""
    return env.unwrapped


def _is_lava(cell) -> bool:
    """Return True if *cell* is a lava tile (None or other types → False)."""
    return cell is not None and getattr(cell, "type", None) == "lava"


def _is_wall(cell) -> bool:
    return cell is not None and getattr(cell, "type", None) == "wall"


def _get_cell(raw_env, pos: np.ndarray):
    """
    Safely retrieve the grid cell at *pos*.
    Returns None when *pos* is out of bounds (treated as wall / non-lava).
    """
    x, y = int(pos[0]), int(pos[1])
    w, h = raw_env.grid.width, raw_env.grid.height
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    return raw_env.grid.get(x, y)


def _front_pos(raw_env) -> np.ndarray:
    """Grid position directly in front of the agent."""
    return np.array(raw_env.agent_pos) + _DIR_TO_VEC[raw_env.agent_dir]


def _turned_dir(current_dir: int, action: int) -> int:
    """
    Compute the agent's direction after a turn action.
    Works for ACTION_LEFT (turn CCW) and ACTION_RIGHT (turn CW).
    """
    if action == ACTION_LEFT:
        return (current_dir - 1) % 4
    if action == ACTION_RIGHT:
        return (current_dir + 1) % 4
    return current_dir


def _adjacent_positions(raw_env) -> list[np.ndarray]:
    """
    Return the four orthogonally adjacent grid positions around the agent
    (front, back, left-side, right-side in absolute grid coords).
    """
    pos = np.array(raw_env.agent_pos)
    return [pos + vec for vec in _DIR_TO_VEC.values()]


# ---------------------------------------------------------------------------
# Public contract functions
# ---------------------------------------------------------------------------

def is_unsafe_action(env: gym.Env, action: int) -> bool:
    """
    Project-contract definition of UNSAFE.

    An action is unsafe if and only if it is ACTION_FORWARD and the cell
    directly in front of the agent is lava.

    Parameters
    ----------
    env    : any gym.Env (wrapper chain is traversed automatically)
    action : integer action index

    Returns
    -------
    bool – True means the action is unsafe and must be blocked.
    """
    if action != ACTION_FORWARD:
        return False
    raw = _unwrapped(env)
    front_cell = _get_cell(raw, _front_pos(raw))
    return _is_lava(front_cell)


def is_risky_action(env: gym.Env, action: int) -> bool:
    """
    Project-contract definition of RISKY (informational, not hard-blocked).

    An action is risky when, after executing it, the agent would be:
      - facing a lava cell  (after a turn), OR
      - adjacent to ≥1 lava cell (after a forward move to a safe cell).

    This function is intentionally conservative: it returns True whenever
    the *post-action* state neighbours lava, so hybrid / old_hybrid-masking
    methods can decide how to penalise or down-weight these actions without
    changing the hard-masking logic.

    Parameters
    ----------
    env    : any gym.Env (wrapper chain is traversed automatically)
    action : integer action index

    Returns
    -------
    bool – True means the action is risky.
    """
    raw = _unwrapped(env)
    pos = np.array(raw.agent_pos)
    cur_dir = raw.agent_dir

    if action == ACTION_FORWARD:
        front = _front_pos(raw)
        front_cell = _get_cell(raw, front)
        # Moving into lava is unsafe, not risky; skip (already handled above).
        if _is_lava(front_cell):
            return False
        # If we can move forward, check whether the new position is adjacent
        # to any lava cell.
        if _is_wall(front_cell):
            return False  # agent won't move; no change in position
        new_pos = front
        for vec in _DIR_TO_VEC.values():
            neighbour = _get_cell(raw, new_pos + vec)
            if _is_lava(neighbour):
                return True
        return False

    if action in (ACTION_LEFT, ACTION_RIGHT):
        new_dir = _turned_dir(cur_dir, action)
        new_front = pos + _DIR_TO_VEC[new_dir]
        return _is_lava(_get_cell(raw, new_front))

    # Pickup / drop / toggle / done: agent stays in place.
    # Mark risky if the agent is already adjacent to lava.
    for vec in _DIR_TO_VEC.values():
        if _is_lava(_get_cell(raw, pos + vec)):
            return True
    return False


def get_nearby_lava_info(env: gym.Env) -> dict:
    """
    Return a dictionary describing lava proximity around the agent.

    Keys
    ----
    front_is_lava     : bool  – cell directly ahead is lava
    left_is_lava      : bool  – cell to the agent's left is lava
    right_is_lava     : bool  – cell to the agent's right is lava
    back_is_lava      : bool  – cell behind the agent is lava
    n_adjacent_lava   : int   – count of orthogonally adjacent lava cells
    would_face_lava_left  : bool – after turning left the agent faces lava
    would_face_lava_right : bool – after turning right the agent faces lava

    This structured output is used by logging hooks and can be directly
    consumed by hybrid / old_hybrid-masking implementations.
    """
    raw = _unwrapped(env)
    pos = np.array(raw.agent_pos)
    cur_dir = raw.agent_dir

    # Cardinal offsets relative to agent direction
    front_vec = _DIR_TO_VEC[cur_dir]
    back_vec  = _DIR_TO_VEC[(cur_dir + 2) % 4]
    left_vec  = _DIR_TO_VEC[(cur_dir - 1) % 4]
    right_vec = _DIR_TO_VEC[(cur_dir + 1) % 4]

    front_lava = _is_lava(_get_cell(raw, pos + front_vec))
    back_lava  = _is_lava(_get_cell(raw, pos + back_vec))
    left_lava  = _is_lava(_get_cell(raw, pos + left_vec))
    right_lava = _is_lava(_get_cell(raw, pos + right_vec))

    # After turning left, what would the agent face?
    new_dir_left  = _turned_dir(cur_dir, ACTION_LEFT)
    new_dir_right = _turned_dir(cur_dir, ACTION_RIGHT)
    face_lava_left  = _is_lava(_get_cell(raw, pos + _DIR_TO_VEC[new_dir_left]))
    face_lava_right = _is_lava(_get_cell(raw, pos + _DIR_TO_VEC[new_dir_right]))

    return {
        "front_is_lava":          front_lava,
        "left_is_lava":           left_lava,
        "right_is_lava":          right_lava,
        "back_is_lava":           back_lava,
        "n_adjacent_lava":        sum([front_lava, back_lava, left_lava, right_lava]),
        "would_face_lava_left":   face_lava_left,
        "would_face_lava_right":  face_lava_right,
    }


def get_action_mask(
    env: gym.Env,
    mask_risky: bool = False,
) -> np.ndarray:
    """
    Compute a boolean action mask for the current environment state.

    Parameters
    ----------
    env         : any gym.Env (wrapper chain is traversed automatically)
    mask_risky  : if True, also mask actions classified as RISKY.
                  Default False (hard masking only blocks UNSAFE actions).
                  Set to True for Method 4 (hybrid masking).

    Returns
    -------
    np.ndarray of shape (N_ACTIONS,), dtype bool
        True  = action is ALLOWED
        False = action is BLOCKED

    Guarantee: at least one entry is always True (safety valve to prevent
    training deadlock when every action would otherwise be masked).

    Usage with sb3-contrib MaskablePPO
    ------------------------------------
        from sb3_contrib.common.wrappers import ActionMasker
        env = ActionMasker(env, lambda e: get_action_mask(e))
        model = MaskablePPO("MlpPolicy", env, ...)
    """
    raw = _unwrapped(env)
    n = raw.action_space.n if hasattr(raw.action_space, "n") else N_ACTIONS
    mask = np.ones(n, dtype=bool)

    for action in range(n):
        if is_unsafe_action(env, action):
            mask[action] = False
        elif mask_risky and is_risky_action(env, action):
            mask[action] = False

    # Safety valve: never return an all-False mask.
    if not mask.any():
        mask[:] = True

    return mask
