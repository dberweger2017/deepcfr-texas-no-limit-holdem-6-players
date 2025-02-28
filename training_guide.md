# DeepCFR Poker Training Guide

This document provides a quick reference for training your Deep Counterfactual Regret Minimization (Deep CFR) poker agent using different methods.

## Training Commands

### Basic Training (vs Random Agents)

Train a new agent against random opponents:

```bash
python train.py --iterations 1000 --traversals 200
```

### Continue Training from Checkpoint

Resume training from a saved checkpoint against random opponents:

```bash
python train.py --checkpoint models/checkpoint_iter_1000.pt --iterations 1000
```

### Self-Play Training

Train against a fixed checkpoint opponent:

```bash
python train.py --checkpoint models/checkpoint_iter_1000.pt --self-play --iterations 1000
```

### Mixed Checkpoint Training

Train against a rotating pool of checkpoint opponents:

```bash
python train.py --mixed --checkpoint-dir models --model-prefix t_ --refresh-interval 1000 --num-opponents 5 --iterations 1000
```

Continue an existing agent with mixed checkpoint training:

```bash
python train.py --mixed --checkpoint models/checkpoint_iter_1000.pt --checkpoint-dir models --model-prefix t_ --iterations 1000
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--iterations` | Number of training iterations | 1000 |
| `--traversals` | Number of game traversals per iteration | 200 |
| `--save-dir` | Directory to save model checkpoints | "models" |
| `--log-dir` | Directory for TensorBoard logs | "logs/deepcfr" |
| `--checkpoint` | Path to checkpoint file to continue training | None |
| `--verbose` | Enable detailed output | False |
| `--self-play` | Train against checkpoint instead of random agents | False |
| `--mixed` | Use mixed checkpoint training | False |
| `--checkpoint-dir` | Directory containing checkpoint models for mixed training | "models" |
| `--model-prefix` | Prefix for models to include in mixed training pool | "t_" |
| `--refresh-interval` | How often to refresh opponents in mixed training | 1000 |
| `--num-opponents` | Number of checkpoint opponents to use in mixed training | 5 |

## Training Strategies

### Random Opponent Training
- Fastest training method
- Good for initial learning
- Agent may overfit to exploit random play

### Self-Play Training
- Trains against a fixed strong opponent
- Helps develop more balanced strategies
- May develop specific counter-strategies to the opponent

### Mixed Checkpoint Training
- Most robust training method
- Prevents overfitting to specific opponent types
- Provides diverse learning experiences
- Closest approximation to Nash equilibrium training

## Monitoring Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.