# Deep CFR Poker Play Guide

This guide will help you play against your trained Deep CFR poker agents using the interactive command-line interface.

## Getting Started

To start a poker game against AI opponents, use one of the following scripts:

- `play_against_models.py` - Play against specifically selected models
- `play_against_random_models.py` - Play against randomly selected models from a directory

## Play Commands

### Playing Against Specific Models

```bash
python play_against_models.py --models models/model1.pt models/model2.pt models/model3.pt
```

### Playing Against Random Models

```bash
python play_against_random_models.py --models-dir models
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--models` | List of specific model paths to use as opponents | [] |
| `--models-dir` | Directory containing model checkpoint files | None |
| `--model-pattern` | File pattern to match model files (e.g., "*.pt") | "*.pt" |
| `--num-models` | Number of models to select randomly | 5 |
| `--position` | Your position at the table (0-5) | 0 |
| `--stake` | Initial chip stack for all players | 200.0 |
| `--sb` | Small blind amount | 1.0 |
| `--bb` | Big blind amount | 2.0 |
| `--verbose` | Show detailed debug output | False |
| `--no-shuffle` | Keep the same random models for all games | False |

## Game Interface

During gameplay, you'll see the current game state displayed, including:

- Your hand cards
- Community cards
- Pot size
- Position of each player
- Current bets
- Available actions

## Available Actions

When it's your turn to act, you can use these commands:

| Command | Action |
|---------|--------|
| `f` | Fold |
| `c` | Check (if available) or Call |
| `r` | Raise (custom amount) |
| `h` | Raise half the pot |
| `p` | Raise the full pot amount |
| `m` | Raise a custom amount |

For raises, you can use the shortcuts above or enter a custom amount when prompted.

## Example Game State

```
======================================================================
Stage: Flop
Pot: $12.00
Button position: Player 3

Community cards: 2♥ K♠ 10♣

Your hand: A♥ K♥

Players:
Player 0 (YOU): $195.00 - Bet: $3.00 - Active
Player 1 (AI): $195.00 - Bet: $3.00 - Active
Player 2 (AI): $200.00 - Bet: $0.00 - Folded
Player 3 (AI): $198.00 - Bet: $2.00 - Active
Player 4 (AI): $199.00 - Bet: $1.00 - Active
Player 5 (AI): $195.00 - Bet: $3.00 - Active

Legal actions:
  c: Check
  r: Raise (min: $3.00, max: $195.00)
    h: Raise half pot
    p: Raise pot
    m: Custom raise amount
======================================================================
```

## Game Progression

1. Each game starts with the dealing of hole cards to all players
2. Betting rounds proceed through preflop, flop, turn, and river
3. After all betting rounds, remaining players show their hands in a showdown
4. Profits/losses are calculated and your balance is updated
5. You'll be asked if you want to continue playing another hand

## Tips for Playing

- Pay attention to the positions - button position rotates each hand
- You can use pot-sized bets (`p`) to put pressure on opponents
- Watch how different models respond to different betting patterns
- Remember that each model might have different strategies based on its training
- Use `--no-shuffle` if you want to study how specific models play over multiple hands

## Advanced Usage

### Playing Against Models of Different Strengths

```bash
# Use only models that have been trained for at least 1000 iterations
python play_against_random_models.py --models-dir models --model-pattern "*_iter_1???.pt" 
```

### Mixing in Random Agents

```bash
# Use only 2 trained models, the rest will be random agents
python play_against_random_models.py --models-dir models --num-models 2
```

### Changing Table Position

```bash
# Play from the button position (typically position 5 in 6-player game)
python play_against_random_models.py --models-dir models --position 5
```

## Analyzing Your Play

After playing against the models, you can analyze your results to understand the strengths and weaknesses of different models and improve your own play against them. The statistics at the end of each session will show your overall performance.