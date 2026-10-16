# v0.5.5 acceptance protocol, version 2: the three-seed blend against Slumbot

**Owner-approved October 16, 2026, before any Slumbot hand of the candidate.** This replaces the [first protocol](hu200-v055-acceptance-protocol.md) for new candidates. It doesn't change that protocol's result: the original recipe was confirmed under its own rule, failed on seed 2026101302 ([#304](https://github.com/dberweger2017/CFR-NLHE-6max/pull/304)), and that failure stays published. Hands from that confirmation, and from any earlier run, never count here.

## Why a new protocol

[#304](reports/hu200-v055-slumbot-confirmation.md) showed two things.

- **Each hand against Slumbot varies by about 16 BB** (standard deviation), and Slumbot deals the cards, so duplicate play can't cancel card luck. At 10,000 hands a 95% interval is about ±32 BB/100.
- **Three separate 10,000-hand tests are a much harder bar than "above −50".** Every seed must measure above about −18. A recipe whose true rate is −10 to −14 passes all three only 22–34% of the time.

This protocol tests one model with more hands, and states its pass chances before play.

**One thing we state openly:** the pooled rule below would have passed the original recipe's pooled 30,000 hands (−13.9 [−32.1, +4.4]). That is exactly why it applies only to fresh hands from a new candidate, declared before any of them is played.

## The candidate

**The frozen three-seed blend that [#306](https://github.com/dberweger2017/CFR-NLHE-6max/pull/306) builds on the 16 GB M4** under the fixed rule of the [blend protocol](hu200-v055-blend-protocol.md). It combines seeds 2026101302, 2026101421 and 2026101422.

- **Its identity:** the model and cache SHA256 recorded in #306's report. The identical bytes must play here.
- **Eligibility:** the candidate is eligible once #306's blend passes its own reproduction, export and audit checks.
- **Internal results:** #306's direct matches against each of the three seeds and the scripted panel are reported with the release, but don't gate this protocol. The owner may decline to start a confirmation after reading them; that decision doesn't change the rule.

**Replication, stated plainly:** the blend is one model built from all three seeds, so there is no second independent blend to confirm it with. The evidence that the blending step itself is sound is #306's comparison of the blend against each of its ingredient seeds. Whether a different set of three seeds would make an equally strong blend is not tested.

## Exploratory play

Exploratory hands of the candidate are allowed only after this protocol is merged. They're reported separately, never pooled with the confirmation, and never change this rule.

## Confirmation

- **Size:** three new dedicated Slumbot accounts (A, B, C), created by the owner. Each account plays its first **10,000** completed hands, so **30,000 hands in total**. No early stop, no extension and no other hands count. The accounts may play in parallel.
- **Policy seeds:** account A uses 20261016055101, B uses 20261016055201 and C uses 20261016055301. If a run stops before 10,000 hands, that account continues in a new run with its next policy seed (…02, …03). Its parts are pooled in play order.
- **Client:** `scripts.play_slumbot` with `--action-translation --translation-max-states 512 --translation-max-events 128`, a 30 s request timeout, and per account `--max-abandoned 200` and `--max-rejected 50`.
- **Validity, per account:**
  - every journal's hash chain verifies;
  - every settlement recomputes from the public actions and revealed cards;
  - Slumbot's session totals reconcile with the verified nets;
  - abandoned hands stay at or below 200 (2%).

  An account that fails validity is reported. Then the confirmation doesn't pass and isn't repeated.

## Rule

Pool the three accounts' 10,000-hand sets (30,000 hands) and compute raw BB/100 with the 95% interval `mean ± 1.96 × SD / √n` over all 30,000 hands, as `scripts.analyze_slumbot_run` reports when given all parts of all three accounts.

- **v0.5.5 qualifies** if the pooled lower bound is above **−50 BB/100**: the blend is not an embarrassing opponent for Slumbot at 200 BB.
- **Reported claims, not gates:**
  - **"beats Slumbot"** only if the pooled lower bound is above 0;
  - **"no detectable loss"** if the interval contains 0;
  - **"loses to Slumbot"** if the upper bound is below 0.
- **Also reported, never deciding:** each account's own 10,000-hand result, all-in-adjusted results, results by seat and the play profile (`scripts.profile_slumbot_runs`).
- **If it doesn't qualify,** the blend isn't released as v0.5.5. All results are published, and the owner decides what comes next.

**Expected behavior:** at 30,000 hands the interval is about ±18.3 BB/100, so the pooled mean must be above about −31.7.

| True rate (BB/100) | Qualifies (above −50) | Lower bound above 0 ("beats Slumbot") |
|---:|---:|---:|
| −50 | 2.5% | 0% |
| −40 | 19% | 0% |
| −30 | 57% | 0% |
| −20 | 90% | 0% |
| −10 | 99% | 0% |
| 0 | >99.9% | 2.5% |
| +10 | >99.9% | 19% |
| +20 | >99.9% | 57% |
| +30 | >99.9% | 90% |

These assume the per-hand spread measured in #304, a standard deviation of about 16 BB per hand. A different spread changes the interval width, not the rule.

## Out of scope

No training, retuning or model change after this protocol merges. The blend's construction is fixed by its own protocol. Pooling with any earlier hands, changing the counts or bar after seeing results, and choosing among several candidates by Slumbot result are all excluded. The release itself needs the owner's go.
