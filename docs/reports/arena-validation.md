# Arena validation protocol

Declared before running the control experiment:

- Use `configs/arena/sensitivity.json`: 128 independent four-player fixed-stack blocks, 100 BB, validation root seed 49021. Run every seat rotation in both arms, for 1,024 hands total.
- Candidate checks/calls; baseline folds whenever legal. Opponents are selected from checks/calls and folds using the saved opponent stream.
- Acceptance: all hands complete legally, accounting reconciles, and the approximate 95% Student-t interval for the paired improvement in BB/100 excludes zero on the positive side.
- This is a deliberately easy infrastructure control. It is not a competitive poker benchmark or evidence about a trained model.
- A two-block smoke run and a comparison of an identical policy against itself must remain inconclusive. Reproducing a saved run must give identical deterministic hand records; timings are excluded.
- If the control is inconclusive or fails, retain the result. Do not extend this budget after inspecting the interval. Diagnose the protocol or implementation in a separate declared experiment.

Results will be recorded below after the checks run.
