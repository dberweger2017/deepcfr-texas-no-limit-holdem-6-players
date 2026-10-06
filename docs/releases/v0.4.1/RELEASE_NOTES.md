# v0.4.1 candidate — opponent-sampled average play

Prepared for owner review; **unpublished**. The candidate uses the exact no-floor, 1B-node opponent-sampled average O export from #165. The first seed was selected prospectively, and all three saved O lineages participate in confirmation.

Fresh audited direct play beats shipped v0.4.0 R1 by **+10.50 [7.90, 13.10] BB/100**. Bounded LBR improves **+37.56 [27.75, 47.37]**, native pressure is **+6.92 [0.39, 13.45]**, and none of the eleven other panels trips the predeclared severe-regression rule. See [#176](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/176) for all panels, absolute values, individual lineages and independent audits. Small panels remain imprecise.

Candidate play is opt-in with `python -m src.play_api.server --o-candidate PATH`. The existing `--policy` command continues to load pinned v0.4.0 B100M. Candidate files are verified with `python -m scripts.verify_v041_model PATH`; the format, native game, legal menu, information boundary and missing/zero-mass fallback are preserved from evaluation.

This is a local two-player 20-BB research preview, with no full-game exploitability or human-strength certification. The bounded attacker still wins against O. Free-sizing human wagers are native legal, while the bot uses its trained menu. Read the model card and inherited engine-license disclosure before redistribution.

No tag, GitHub release, Latest change or default-model replacement is authorized by this preparation. After explicit owner approval, regenerate the publication manifest against the approved merged source commit and rerun the required checks; never use a pre-merge preparation SHA as the release target.
