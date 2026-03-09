# Future Improvements for Deep CFR Poker

This file tracks ideas that still make sense after the recent architecture changes and bug-fix work on `main`.

Items removed from the old version were mostly already done or no longer the right priority. In particular, the repo already has:

- continuous raise sizing
- prioritized replay for advantage memory
- opponent-history modeling
- self-play and mixed-training path fixes
- targeted regression tests for recent `pokers` and training failures

## 1. Reproducibility and Evaluation

- [ ] Publish a repeatable benchmark recipe with fixed seeds, checkpoint schedules, and evaluation games
- [ ] Add stronger evaluation baselines than random opponents alone
- [ ] Build a standard tournament suite for comparing checkpoints across training phases
- [ ] Track variance, not just mean profit, when reporting model quality

## 2. Training Scale and Throughput

- [ ] Parallelize traversal data collection so larger training runs are practical
- [ ] Batch more model inference inside traversals to reduce overhead
- [ ] Move more hyperparameters into explicit config files instead of hard-coded defaults
- [ ] Add lightweight experiment tracking for seeds, losses, profits, and checkpoint metadata

## 3. Algorithmic Improvements

- [ ] Evaluate discounted CFR / linear CFR style weighting more systematically
- [ ] Add suit-isomorphism or related state augmentation to improve sample efficiency
- [ ] Improve checkpoint-pool selection for mixed training so the opponent set stays diverse and informative
- [ ] Test larger or alternative network bodies only after reproducibility is under control

## 4. Opponent Modeling

- [ ] Run ablations to verify whether opponent modeling improves EV over the simpler baseline
- [ ] Improve the way opponent features are injected into action and sizing decisions
- [ ] Explore table-image style features so opponents can react to the agent's own recent behavior

## 5. Product and Developer Experience

- [ ] Clean up packaging so editable installs and console entry points are first-class
- [ ] Improve documentation around which workflows are stable versus experimental
- [ ] Add a small evaluation CLI that loads checkpoints and prints comparable metrics without editing scripts
- [ ] Revisit model export and deployment only after the training pipeline is stable and reproducible
