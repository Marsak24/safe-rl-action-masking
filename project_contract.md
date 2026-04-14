# Project Contract (Draft to be discussed)

## Environment & Safety

Function to define unsafe actions:

def is_unsafe_action(obs, action) -> bool

- Returns True if the action leads to lava (unsafe)
- Returns False otherwise

---
## Action Masking
Function to generate mask:

def get_action_mask(obs) -> list

- Returns a list of booleans (or 0/1)
- Same length as action space
- 0 = action not allowed (unsafe)
- 1 = action allowed

---

## Reward Modification

Penalty function:

def apply_penalty(reward, unsafe, penalty_value)

- If unsafe = True → subtract penalty
- Otherwise → keep reward

---
## Logging Format
Each episode should log:
reward
success (0 or 1)
violations (count)
episode length
visited states (optional for heatmap)

---

## Definition of Safety Violation
We define a violation as:

- The agent enters a lava cell

(Optional extension: also count attempted unsafe actions)

---

## Shared Rules

- All methods must use the SAME environment
- All methods must use the SAME hyperparameters
- Only safety mechanism changes between methods
