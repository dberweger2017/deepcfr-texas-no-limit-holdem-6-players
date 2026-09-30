# GPT-6 Luna browser-play protocol

Frozen before any poker outcome on September 30, 2026. This exploratory
LLM-player experiment uses the public `v0.4.0` source tag at
`a2053bbeea8a5e5170a1d83bf5d440684f82283d` and the public fixed-first-seed
B100M inference export: 40,144,034 compressed bytes, SHA-256
`4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf`.
The game is two-player 20BB no-limit Hold'em, no rake/ante, with 2,000 chips
reset per seat every hand. Source/model identity is checked before loading.

The requested player is **`gpt-6-luna`, high reasoning**. The accepted runtime
configuration must be recorded; an unavailable requested model stops the
experiment. The M1 runs one loopback service and the Codex internal browser.
I have authorized B100M inference on M1 for this task. M4 is not used.

The [player prompt](luna-browser-player-prompt.txt) and
[technical harness](luna-browser-harness.txt) are frozen and hashed before the
preflight. Every child is created without inherited conversation history.
The child can use only rendered table observations and computer-use actions.
Tool restrictions are instructional because the sub-agent API does not expose
a tool allowlist; the complete tool-call transcript must be audited after each
session. A prohibited tool or hidden-information exposure invalidates the run
and stops it. Private reasoning is neither requested nor published.

Run order: a separate 10-hand restricted preflight; a new persistent Luna
context for exactly 500 restricted B100M hands; then a new context and the same
prompt for 100 hands against a separately identified uniform-random opponent.
The random adapter is added only after the primary run and samples uniformly
from the same concrete restricted legal menu with a separate private RNG. No
primary policy or actual native wager is changed. All sessions use the
existing benchmark-safe visibility and alternating seat-0-button-first
schedule. No result-based stopping, coaching, context reset or unfavorable
result removal is allowed. Technical failures remain failed/incomplete.

Record visible attempted actions, legal labels, accepted native actions,
latency, retries, tool/UI errors and technical interventions. Reconcile every
decision with the journal after the session; private records remain outside
Git. Publish sanitized exports and action/tool metadata only. After completion
verify exact hand counts, positions, settlement, public event digests and all
native replays, and audit the transcript for prohibited tools/information.
Report each opponent separately with raw chips, BB, BB/100, position results,
wins/losses/ties, terminal pots, B100M trained/fallback counts, latency
mean/median/p95 and resource/usage measures that the environment surfaces.
No confidence interval or formal strength qualification is claimed.

After all results are frozen, review the first ten hands, every 50th hand and
the five largest positive and negative terminal payoffs per session, with
ordinal tie-breaking and deduplication. Describe only observable actions and
computer-use failures. Preserve unfavorable completed results. Provide one
recommendation for a later experiment; do not run another model or extend the
random session without approval. The final focused PR remains draft.
