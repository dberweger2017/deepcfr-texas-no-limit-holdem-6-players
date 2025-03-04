# Deep CFR Poker AI Training Journey

This document chronicles my journey developing and training a Deep Counterfactual Regret Minimization (CFR) poker AI for 6-player No-Limit Texas Hold'em. The training process evolved through multiple methodologies, ultimately producing a model capable of defeating both random opponents and sophisticated AI models.

## Training Philosophy and Evolution

I began my Deep CFR poker AI training process with three parallel models (100, 200, and 400 traversals per iteration), discovering that higher traversal counts yielded better performance but at significant computational cost. After observing that the 400-traversal model consistently outperformed others against random opponents, I continued with self-play training for 7000+ iterations, which produced interesting cyclic patterns but ultimately failed to surpass the checkpoint model it trained against. I noted that performance peaked at different iteration points (2000-2500 and 5000-5500) against random opponents, suggesting strategic exploration rather than linear improvement. 

Finally, I implemented mixed training using a diverse opponent pool from my previous self-play checkpoints, reduced the learning rate by half for more stable learning, and observed dramatic improvements with faster adaptation to diverse strategies. The key insight was that efficient training benefits from a phased approach: starting with fewer traversals for basic strategy learning, increasing traversals for refinement, and ultimately exposing the model to diverse opponents to develop robust, unexploitable poker strategies that generalize beyond a single opponent type.

## Training Phases and Commands

### Phase 1: Initial Training with Different Traversal Counts
Experimenting with different traversal counts to find the optimal exploration-exploitation balance:

```bash
# Training with 100 traversals per iteration
python train.py --iterations 1000 --traversals 100 --log-dir logs/100 --save-dir models/100

# Training with 200 traversals per iteration
python train.py --iterations 1000 --traversals 200 --log-dir logs/200 --save-dir models/200

# Training with 400 traversals per iteration
python train.py --iterations 1000 --traversals 400 --log-dir logs/400 --save-dir models/400
```

### Phase 2: Extended Training of Best Model
Continuing training of the 400-traversal model to develop deeper strategic understanding:

```bash
python train.py --checkpoint models/400/checkpoint_iter_1000.pt --iterations 1000 --traversals 400 --log-dir logs/400 --save-dir models/400
```

### Phase 3: Self-Play Training
Training the model against a fixed version of itself to develop more sophisticated play:

```bash
# First round of self-play (400 traversals)
python train.py --checkpoint models/400/checkpoint_iter_2000.pt --self-play --iterations 2000 --traversals 400

# Lower traversal self-play for faster iteration
python train.py --checkpoint models/selfplay_checkpoint_iter_2000.pt --self-play --iterations 2000 --traversals 100

# Extended self-play with higher traversals
python train.py --checkpoint models/selfplay_checkpoint_iter_2000.pt --self-play --iterations 2000 --traversals 400 --log-dir logs/selfplay2 --save-dir selfplay2
```

### Phase 4: Mixed Training
Training against a diverse pool of opponents with periodically refreshed selection:

```bash
# Mixed training with half learning rate (0.00005)
python train.py --mixed --checkpoint-dir selfplay3 --model-prefix selfplay --iterations 20000 --traversals 400 --log-dir logs/mixed --save-dir models/mixed --refresh-interval 1000
```

## Key Insights and Results

### Traversal Counts
- Higher traversals (400+) led to better performance but slower training
- For early training phases, lower traversals (100) were more efficient for developing basic strategies
- Optimal approach: start with low traversals, then increase as training progresses

### Self-Play Dynamics
- Self-play training produced interesting cyclic patterns in model performance
- The model never consistently beat its training opponent (fixed checkpoint model)
- Performance against random opponents plateaued around 15-17 chips profit per game
- Strategic cycles suggested exploration of different strategic spaces rather than convergence

### Mixed Training Breakthrough
- Mixed training against diverse opponents produced dramatic improvements
- Within 3000 iterations, the mixed model achieved what took self-play 8000+ iterations
- Performance against random opponents peaked at 20+ chips profit
- Most significantly, the model began to defeat the mixed opponent pool it trained against

## Conclusions

The most effective training methodology emerged as a three-stage process:
1. Initial training against random opponents to develop fundamental poker concepts
2. Self-play to develop more sophisticated strategic thinking
3. Mixed training against diverse opponents to develop robust, generalizable strategies

This progressive approach mirrors techniques used in cutting-edge AI research and produced a poker AI capable of defeating both random players and trained models. The final model shows promise for competing in real-world low-stakes online poker environments.

The training results highlight the importance of diverse training environments in developing robust AI for imperfect information games. The significant performance improvements achieved through mixed training suggest that exposure to varied strategic approaches is crucial for developing truly strong poker AI.

Im thinking very about possible consequences before publishing the final trained model here. Feel free to dm me in private if you want to discuss this further. Or want to test the model in a private environment.