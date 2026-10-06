# Luna versus 0.4.0-shield: completed Chrome match

On October 6, 2026, the owner's separately launched **GPT-6 Luna / HIGH** session completed **400/400 hands** against **0.4.0-shield / 0.4.0-s** in native Chrome. Luna lost **82 BB**, or **20.50 BB/100**; Shield won the corresponding 82 BB. All 400 hands independently replay against the pinned native engine and model identity.

This is a **replay-verified descriptive match with a failed browser protocol audit**. The frozen player text was not present in the actual player's user messages, the planned browser attempt/observation records were absent, and tool use exceeded the prepared browser-only boundary. The strict reporter rejects this run; its gates remain intact. Native replay verifies accepted actions and settlements, but cannot substitute for the missing browser intent, retry and latency evidence.

## Result and identity

| Luna's seat | Hands | Net BB | BB/100 |
|---|---:|---:|---:|
| Button / small blind | 200 | −120.5 | −60.25 |
| Big blind | 200 | +38.5 | +19.25 |
| Total | 400 | −82.0 | −20.50 |

Luna recorded 180 positive-payoff hands, 217 negative-payoff hands and 3 ties. Net chips were −8,200; average pot size was 8.075 BB. Session creation was **10:57:31 Madrid**, completion **13:33:44**, an elapsed **2h 36m 13s including setup**. Creation is not a measured first-deal timestamp.

Runtime `turn_context` records confirm `gpt-6-luna` with `high` effort. Shield is #171's first-lineage CFR+ traverser-reach average, seed 2026100601, 1B training nodes. Policy SHA256: `a5e9d0fc6f4a448640f52f508187f779e43a0a8fd41158207f03de82adc219e1`. Runtime game source: `2fa67013dd46355dfa996133dc19afef21f0d4c6`. The [frozen configuration](../../configs/diagnostics/luna-shield-chrome.json) and [preparation protocol](../luna-shield-chrome.md) remain available as the intended protocol, not a claim that the actual player followed it.

The journal contains **1,012 accepted human decisions**, **938 accepted bot decisions**, **938 trained lookups**, **zero fallback lookups** and zero off-menu actions. Replay checks chip conservation, each payoff, alternating positions and all public-event digests; the saved final HTTP export agrees with journal totals and identities. The owned service and sampler stopped after final export and replay. Chrome control remained with the independent player after preparation.

## Comparison with Luna's earlier v0.4.0 match

Luna's result changed from **+107 BB** against v0.4.0 B100M to **−82 BB** against Shield over 400 completed hands each: a **189 BB swing**, with **175 BB** coming from the button/small blind. Both runs include twelve full-stack wins; this run adds three full-stack losses. The [full comparison](luna-shield-v040-comparison.md) decomposes large-call exposure and three-bet continuations and records the changed deals, player context, stopping rule and browser audit. This is descriptive matchup evidence, not a paired estimate of relative strength or demonstrated learning.

## Why the persistent session is useful

Luna can retain earlier public observations in its conversation context and try to identify how Shield plays. That makes this session useful for exploring opponent tendencies and possible adaptation across hands, beyond isolated decisions. Luna's [verbatim post-session retrospective](luna-shield-chrome/luna-retrospective.md), supplied by the owner, reports small probes, respect for larger bets and difficulty continuing against three-bets from the button/small blind.

Those observations are hypotheses for reviewing the public history. They do not demonstrate statistically improved play, reliable exploitation or a positive expected value. The final loss and position split do not by themselves measure learning either. No controlled adaptation comparison was run, and the tool-boundary failures prevent treating this as a clean browser-only benchmark. The historical #127 match used different deals and a different opponent; the raw scores cannot establish a causal strength improvement. No model promotion, release or general strength claim follows.

## Checking the retrospective's examples

| Hand | Publicly supported events | Qualification |
|---|---|---|
| 378 | K♦Q♥, straight on the J♥ turn; river shove called; +20 BB | Shield **mucked**. Its cards and the claimed queen-high holding are unknown in the public record. The quoted claim is unverified and must not support a calling-tendency conclusion. |
| 390 | A♦5♥; 1 BB turn bet wins +4 BB | Luna acted first on the turn; this example is not itself a probe after a bot check. |
| 387 | 6♠3♠; three-bet continuation and flop call; −5 BB; Shield showed Q♣J♣ for top pair | The flop call had a flush draw. A loss alone does not establish that a call was an expected-value error. |
| 393 | A♥J♥; three-bet call, checked-back gutshot flop, fold to a 12 BB turn bet; −6 BB | The action sequence and payoff agree with the retrospective; optimal play is not established by the terminal result. |

These four examples come from Luna's retrospective. The separate [frozen qualitative selection](luna-shield-chrome/qualitative-selection.json) retains the first ten hands, every 50th hand, and five largest positive and negative payoffs, with ordinal tie-breaking and deduplication. Neither selection supplies action-value ground truth. Only legitimately revealed opponent cards appear in the published history; mucked cards are not recovered for publication.

## Browser audit qualification

There were **1,914 CUA calls and four `exec` calls**, with **zero `luna_attempt` and zero `luna_observed` records**. The frozen-boundary audit flags **13 invocations**: seven surface inventories, one tab inventory, one attempted table binding with the mistyped `172.0.0.1` origin, three pre-play shell/repository reads, and one post-completion goal update. The repository reads included documentation and source searches; the post-completion orchestration call is distinguished from information access during play. The parser also detected two existing-tab lookup failures. Those detections are not a complete retry/error count.

The exact prepared prompt/logging text was absent from the player user messages, so this is also a handoff failure; it is not evidence that Luna knowingly ignored those instructions. No hidden-deal access was confirmed, but source access prevents establishing the intended information boundary. Attempt/accepted mismatches, retry counts and per-decision browser latency remain **unverified**, rather than being reported as zero. The original rollout and sanitized tool-code hash ledger remain private; no private reasoning is published.

## Evidence and validation

- [Actual final HTTP export](luna-shield-chrome/export.json), [native hand CSV](luna-shield-chrome/native-hands.csv) and [native accepted-decision CSV](luna-shield-chrome/native-decisions.csv).
- [Public history](luna-shield-chrome/public-history.json), [audit summary](luna-shield-chrome/audit.json), [resource summary](luna-shield-chrome/resources.json) and [evidence hashes](luna-shield-chrome/manifest.json).
- All 400 hands replayed; export, wins/losses/ties, position splits, accepted-decision counts and public digests independently recomputed. The strict browser report's rejection is retained locally.
- After integration with current main, the same 57 focused play/audit/average/release tests pass, and the public-record comparison reproduces exactly.
- Preparation passed 57 focused play/audit/average/release tests, JavaScript syntax and diff checks, plus a separate two-hand real-policy smoke. The report adds no game behavior changes.

The sampler retained 318 snapshots. Maximum observed service RSS was **2,208,368 KiB (2.106 GiB)** and CPU 28.2%; these are sampled maxima, not guaranteed lifetime peaks. Host swap ranged from 2,809.94 to 7,125.19 MiB and includes unrelated applications. Private journals, access credentials, raw resource samples and the player rollout stay outside Git and remain locally retained under the [artifact index](../../RESULTS_INDEX.md).
