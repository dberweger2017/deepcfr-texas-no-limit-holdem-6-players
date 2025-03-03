# Deep CFR Poker AI Training Journey

I began my Deep CFR poker AI training process with three parallel models (100, 200, and 400 traversals per iteration), discovering that higher traversal counts yielded better performance but at significant computational cost. After observing that the 400-traversal model consistently outperformed others against random opponents, I continued with self-play training for 7000+ iterations, which produced interesting cyclic patterns but ultimately failed to surpass the checkpoint model it trained against. I noted that performance peaked at different iteration points (2000-2500 and 5000-5500) against random opponents, suggesting strategic exploration rather than linear improvement. Finally, I implemented mixed training using a diverse opponent pool from my previous self-play checkpoints, reduced the learning rate by half for more stable learning, and observed promising early results with faster adaptation to diverse strategies. The key insight was that efficient training benefits from a phased approach: starting with fewer traversals for basic strategy learning, increasing traversals for refinement, and ultimately exposing the model to diverse opponents to develop robust, unexploitable poker strategies that generalize beyond a single opponent type.

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

Through these progressive training phases, the model developed from basic poker skills to more nuanced strategic play, with each phase building upon the knowledge gained in previous stages.