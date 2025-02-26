# DeepCFR Poker AI

A deep learning implementation of Counterfactual Regret Minimization (CFR) for No-Limit Texas Hold'em Poker. This project demonstrates advanced reinforcement learning techniques applied to imperfect information games.

![Poker AI](https://raw.githubusercontent.com/dberweger2017/deepcfr/refs/heads/main/images/profit.png)

## Overview

This repository implements Deep Counterfactual Regret Minimization (Deep CFR), an advanced reinforcement learning algorithm for solving imperfect information games. The implementation focuses on No-Limit Texas Hold'em Poker, one of the most challenging games for AI due to:

- Hidden information (opponents' cards)
- Stochastic elements (card dealing)
- Sequential decision-making
- Large state and action spaces

The agent learns by playing against random opponents and self-play, using neural networks to approximate regret values and optimal strategies.

## Architecture

The implementation consists of three main components:

1. **Model Architecture** (`model.py`)
   - Neural network definition
   - State encoding/representation
   - Forward pass implementation

2. **Deep CFR Agent** (`deep_cfr.py`) 
   - Advantage network for regret estimation
   - Strategy network for action selection
   - Memory buffers for experience storage
   - CFR traversal implementation
   - Training procedures

3. **Training Pipeline** (`train.py`)
   - Training loop implementation
   - Evaluation against random agents
   - Metrics tracking and logging
   - Model checkpointing and saving

## Technical Implementation

### Neural Networks

The project uses PyTorch to implement:

- **Advantage Network**: Predicts counterfactual regrets for each action
- **Strategy Network**: Outputs a probability distribution over actions

Both networks share a similar architecture:
- Fully connected layers with ReLU activations
- Input size determined by state encoding (cards, bets, game state)
- Output size matching the action space

### State Representation

Game states are encoded as fixed-length vectors that capture:
- Player hole cards (52 dimensions for card presence)
- Community cards (52 dimensions)
- Game stage (5 dimensions for preflop, flop, turn, river, showdown)
- Pot size (normalized)
- Player positions and button
- Current player
- Player states (active status, bets, chips)
- Legal actions
- Previous actions

### Action Space

The agent can select from four strategic actions:
- Fold
- Check/Call
- Raise 0.5x pot
- Raise 1x pot

### Counterfactual Regret Minimization

The implementation uses external sampling Monte Carlo CFR:
- Regret values are used to guide exploration
- Strategy improvement over iterations
- Regret matching for action selection
- Importance sampling for efficient learning

### Training Process

The training procedure includes:
- Data generation through game tree traversal
- Experience collection in memory buffers
- Regular network updates
- Periodic strategy network training
- Regular evaluation against random opponents
- Progress tracking via TensorBoard

## Performance Optimizations

The implementation includes various optimizations:
- Gradient clipping to prevent exploding gradients
- Huber loss for robust learning with outliers
- Regret normalization and clipping
- Linear CFR weighting for faster convergence
- Efficient memory management

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- TensorBoard
- [Pokers](https://github.com/yourusername/pokers) (custom poker environment)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deepcfr-poker.git
cd deepcfr-poker

# Install dependencies
pip install -r requirements.txt

# Install the poker environment
pip install pokers
```

## Usage

### Training

```bash
# Basic training
python train.py

# With custom parameters
python train.py --iterations 2000 --traversals 500 --save-dir models/experiment1 --log-dir logs/experiment1
```

### Monitoring

```bash
# Start TensorBoard
tensorboard --logdir=logs/deepcfr

# Then open http://localhost:6006 in your browser
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py --model models/deep_cfr_iter_1000.pt --games 1000
```

## Results

After training, the agent achieves:
- Positive expected value against random opponents
- Increasing performance over training iterations
- Sophisticated betting strategies

![Learning Curve](https://github.com/yourusername/deepcfr-poker/assets/yourusername/learning-curve.png)

## Future Work

- Implement opponent modeling
- Expand the action space with more bet sizing options
- Experiment with alternative network architectures (CNN, Transformers)
- Parallel data generation for faster training
- Develop a more diverse set of evaluation opponents

## References

1. Brown, N., & Sandholm, T. (2019). [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164). *ICML*.
2. Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2008). [Regret Minimization in Games with Incomplete Information](https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf). *NIPS*.
3. Heinrich, J., & Silver, D. (2016). [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121). *arXiv preprint*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The [Annual Computer Poker Competition](http://www.computerpokercompetition.org/) for inspiration
- [OpenAI](https://openai.com/) and [DeepMind](https://deepmind.com/) for pioneering work in game AI
- The PyTorch team for their excellent deep learning framework