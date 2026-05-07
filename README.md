# Safe Reinforcement Learning with Action Masking

## Overview

This project investigates the trade-off between **safety and exploration** in reinforcement learning. We compare multiple PPO-based approaches for enforcing safety in constrained environments using the [MiniGrid](https://minigrid.farama.org/) LavaGap and LavaCrossing benchmarks.

![LavaGap S7](lavagap_s7.gif)

We implement and evaluate six approaches:

| Approach | Description |
|---|---|
| **Vanilla PPO** | Baseline with no safety mechanism |
| **Penalty PPO** | Negative reward on lava contact |
| **Penalty PPO (adjacent)** | Penalty extended to cells adjacent to lava |
| **Hard Action Masking** | Uses `MaskablePPO`; illegal actions are zeroed out of the policy |
| **Soft Action Masking** | Safety logit penalties applied at action selection |
| **Hybrid Masking** | Combines hard masking with a penalty signal |

The goal is to understand how these mechanisms affect:
- Learning performance and convergence speed
- Safety violations (lava contacts)
- Exploration coverage
- Generalization to unseen environments

---

## Requirements

- Python 3.8+
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) (for `MaskablePPO`)
- [minigrid](https://minigrid.farama.org/)
- gymnasium, numpy, matplotlib, pandas, torch, pygame

---

## Installation

```bash
git clone https://github.com/Marsak24/safe-rl-action-masking
cd safe-rl-action-masking
pip install -r requirements.txt
```

---

## Project Structure

```
safe-rl-action-masking/
├── agents/       # Training scripts for each method
├── env/          # Environment wrappers (masking, penalties, logging)
├── metrics/      # Evaluation, logging, and plotting utilities
├── results/      # Experiment outputs (CSV logs, plots, videos)
├── src/          # Core masking utilities
└── tests/        # Unit tests
```

---

## Training

### Vanilla PPO

```bash
python agents/train_ppo_multi_env_logging.py
```

### Penalty PPO

```bash
python agents/train_penalty.py
```

### Hard Action Masking (MaskablePPO)

```bash
python agents/train_masked_ppo.py
```

### Hybrid Masking

```bash
python agents/train_hybrid_ppo.py
```

### Hybrid Adaptive Masking

```bash
python agents/train_hybrid_adaptive_ppo.py
```

> **Note:** Each training script runs across multiple random seeds and saves logs, models, and videos under `results/`.

---

## Evaluation

Evaluate a trained agent on held-out environments:

```bash
python metrics/evaluation.py
```

Evaluate without masking (mask removal test):

```bash
python metrics/eval_without_mask.py
```

Plot aggregated results across all methods:

```bash
python metrics/plot_results.py
```

---

## Outputs

Each run produces:

- Training logs (CSV) per seed and aggregated summaries
- Evaluation metrics: reward, success rate, lava violations
- Learning curve and convergence plots
- Exploration heatmaps
- Recorded videos of agent behavior

Results are organized under `results/<method>/`.

---

## Environments

**Primary (training):**
- `MiniGrid-LavaGapS5-v0`
- `MiniGrid-LavaGapS6-v0`
- `MiniGrid-LavaGapS7-v0`

**Generalization (held-out):**
- `MiniGrid-LavaCrossingS9N1-v0`
- `MiniGrid-LavaCrossingS9N3-v0`
- `MiniGrid-LavaCrossingS11N5-v0`

---

## Results

Full training logs, plots, and videos are available here:
[Detailed Training Results and Plots](https://mailaub-my.sharepoint.com/:f:/g/personal/shi12_mail_aub_edu/IgBuB5hg8Z3NQZLDu0Gp4B0VAYFsbAECKZFIwUh6kWbEHfk?e=JecLBw)

---

## Authors

- Marwah Al Sakkaf
- Sara Ibrahim
- Rawan Darwich
- Haifa Naim