# GPT-6 Luna browser poker experiment

## Preflight — completed September 30, 2026

I ran a separate ten-hand preflight before the planned primary experiment.
The runtime confirmed `gpt-6-luna` with `high` reasoning. It used only rendered
table observations and browser controls, with no inherited parent conversation.
The frozen [protocol](../luna-browser-benchmark-protocol.md) records the prompt,
game, model hash, run order and sampling rules.

| Check | Preflight evidence |
| --- | --- |
| Completed hands | 10 / 10; five button/SB, five BB |
| Player net | 0 chips / 0 BB / 0 BB per 100 |
| Wins / losses / ties | 3 / 7 / 0 |
| Native replay | All ten settlement amounts, chip conservation and public digests matched |
| Human decisions | All 32 attempted actions reconciled with accepted native actions |
| Bot lookup | 31 trained lookups, zero fallback; inspected only after completion |
| Browser tools | 56 CUA calls; no prohibited tool or capability detected |
| Parent poker intervention | None |

The initial existing-tab lookup failed, and this child's browser did not support
the requested visible-tab option. Luna recovered by opening the same local table
in its browser. No subsequent UI error was reported. Tool restrictions are
instructional, not an enforced tool allowlist; the transcript audit is required.
I do not publish private reasoning or the unfiltered model transcript.

### Raw evidence

- [Sanitized HTTP benchmark export](luna-browser/preflight/export.json).
- [Every human decision](luna-browser/preflight/decisions.csv): visible menu and
  attempted button, accepted native action/target, timestamps and UI confirmation.
- [Per-hand results and public digests](luna-browser/preflight/hands.csv).
- [Tool, configuration, replay and usage summary](luna-browser/preflight/audit.json).

UI acknowledgment averaged 298 ms (median 300 ms, p95 385 ms). This measures
attempt-to-observation time, **not** reasoning latency. The current table does
not log the instant when a decision first becomes visible; total decision latency
will be reported separately with its measurement definition.

The service's initial sampled RSS was 1,065,728 KiB (1.016 GiB), followed by a
much lower idle resident sample. These are snapshots, not a measured peak or
total logical memory footprint; macOS compression/pageout can reduce RSS.
The longer runs will collect periodic service RSS/CPU and system memory/swap
samples. M1 has 16 GiB RAM. M4 is not used.

Usage counters reported 3,772,472 input tokens (3,644,416 cached), 11,017 output
tokens and 2,623 reasoning-output tokens. These are environment counters, not
published reasoning or a calculated monetary charge.

## Primary and calibration — pending

The 500-hand restricted B100M session will use a fresh, persistent Luna context.
Only after it finishes will I add and test the separately identified uniform
restricted-menu random opponent, then run a fresh 100-hand context. Results will
remain separate. This ten-hand preflight is integration evidence, not a playing
strength estimate or a substitute for either planned session.

## Reproduction and audit

Use the public `v0.4.0` source/model and the [local launch instructions](../play-web.md).
Freeze and hash the linked prompt/harness before outcomes, create the exact-N
restricted benchmark, and dispatch the named model with no inherited context.
After completion, run `python -m scripts.audit_luna_browser ROLLOUT --out PRIVATE_AUDIT`
and `python -m scripts.report_luna_browser PRIVATE_DB ROLLOUT TOKEN_FILE OUTPUT_DIR`.
The report command requires one isolated completed session and the running local
service; it fetches the actual sanitized HTTP export. Private database, token and
rollout remain outside Git. CSVs intentionally omit cards, seeds, RNG state,
private policy keys and private reasoning. Failed/incomplete runs must be retained.
