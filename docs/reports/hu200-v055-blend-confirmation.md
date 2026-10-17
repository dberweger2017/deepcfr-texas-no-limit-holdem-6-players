# v0.5.5 confirmation: the three-seed blend qualifies against Slumbot

**v0.5.5 qualifies.** Under the owner-approved [protocol version 2](../hu200-v055-acceptance-protocol-v2.md), declared before any of the candidate's hands, [#306](https://github.com/dberweger2017/CFR-NLHE-6max/pull/306)'s audited three-seed blend played 10,000 fresh hands on each of three new Slumbot accounts. The pooled 30,000-hand 95% lower bound is **−38.57 BB/100**, above the −50 bar.

| Pooled raw, 30,000 hands | BB/100 [95%] |
|---|---|
| **Result** | **−19.67 [−38.57, −0.76]** |

**The protocol's declared claims:**
- **The blend is not an embarrassing opponent for Slumbot at 200 BB:** the lower bound is above −50.
- **It loses to Slumbot:** the upper bound, −0.76, is below 0. The loss is small, around 20 BB/100, but detectable at 30,000 hands.
- **It does not beat Slumbot.**

The release itself needs the owner's go.

## Runs and validity

The candidate was #306's audited blend average, SHA256 `ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485`, identical bytes to #306's manifest. It played with its verified read-only policy cache, the pinned K50 tables, translation 512/128, a 30 s timeout and per-account budgets of 200 abandoned and 50 rejected hands, at client commit `e2f3de1c`. The three accounts played in parallel on cygnus-x1 from 22:02 Madrid on October 16.

| Account | Policy seed | Hands | Abandoned | Rejected | Session reconciles | BB/100 [95%] |
|---|---|---:|---:|---:|---|---|
| A, `deepcfr-v055-ba` | 20261016055101 | 10,000 | 0 | 0 | yes, 0 unverified chips | −24.29 [−57.34, +8.76] |
| B, `deepcfr-v055-bb` | 20261016055201 | 10,000 | 0 | 0 | yes, 0 unverified chips | −23.75 [−55.72, +8.22] |
| C, `deepcfr-v055-bc` | 20261016055301 | 10,000 | 0 | 0 | yes, 0 unverified chips | −10.97 [−44.17, +22.23] |

- **Journals:** every journal's hash chain verifies.
- **Sessions:** each account played one uninterrupted session; Slumbot's session totals equal our verified nets.
- **Clean runs:** no run stopped early, so no account needed a second part.
- **Separation:** exploratory hands (account `deepcfr-v055-bx`) are reported separately and never pooled here.

## Reported alongside, not deciding

| | BB/100 [95%] |
|---|---|
| All-in adjusted | −23.77 [−41.54, −6.01] |
| Big blind | −32.8 [−59.5, −6.1] |
| Button | −6.5 [−33.3, +20.3] |

**Where the loss comes from.** Chips won or lost per hand, by how hands ended (BB/100 over all 30,000 hands):

| Ending | Our folds | Slumbot's folds | Net |
|---|---:|---:|---:|
| Preflop | 5,267 hands, −21.2 | 4,649 hands, +19.8 | −1.4 |
| **Flop** | **3,775 hands, −38.7** | **2,795 hands, +25.8** | **−12.9** |
| Turn | 2,154 hands, −35.5 | 1,857 hands, +34.7 | −0.8 |
| River | 1,793 hands, −46.9 | 1,683 hands, +52.1 | +5.2 |
| Showdown | 5,983 hands | | −7.1 |
| All-in before river | 44 hands | | −2.7 |

**Two weaknesses stand out:**
- **On the flop, the blend folds about 35% more often than Slumbot** (3,775 against 2,795 hands), and that exchange costs about 13 BB/100: most of the loss.
- **The big blind loses clearly** (−32.8, interval excluding zero), consistent with the big-blind deficit [AIVAT](slumbot-aivat-real-runs.md) found in every earlier run.

River play, by contrast, gains about 5 BB/100.

## In context

| Run | Hands | Raw BB/100 [95%] |
|---|---:|---|
| #304 seeds pooled (v0.5.5 protocol 1, failed on one seed) | 30,000 | −13.87 [−32.14, +4.40] |
| Blend, exploratory (`-bx`, not pooled) | 10,000 | −6.79 [−39.65, +26.08] |
| **Blend, confirmation (this report)** | **30,000** | **−19.67 [−38.57, −0.76]** |

All three estimates are consistent with one strength of roughly −10 to −20 BB/100. #306 measured the blend only about +2 BB/100 stronger than its seeds internally. The confirmation's lower mean than the exploratory run is within noise.

## Files

The three journals, summaries, launchers and the pooled analysis are archived; see [RESULTS_INDEX](../../RESULTS_INDEX.md). The per-account summaries and the pooled analysis are in [`hu200-v055-blend-confirmation/`](hu200-v055-blend-confirmation/). Reproduce the pooled row with `python -m scripts.analyze_slumbot_run ACCOUNT_A ACCOUNT_B ACCOUNT_C --out pooled.json`.

AIVAT on these hands, for information only, can follow with the tools from [#316](slumbot-aivat-real-runs.md). It is not part of the declared rule.
