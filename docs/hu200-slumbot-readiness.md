# Remaining Slumbot integration after HU200 support

[PR213's assessment](hu100-external-benchmark.md) remains the source for observed
external access and the decision-prefix codec. This task closes fixed-depth
identity admission for a genuinely trained HU200 model. It makes no new service
request. v0.5.0 still targets internally evaluated HU100; HU200 evidence belongs
to v0.5.5 and does not establish HU100 strength.

The existing `SlumbotAdapter` supplies public-only observations, validates legal
street-local raise-to actions, and strips tokens, opponent cards and evaluator
winnings from policy inputs. `SlumbotConnection` has injected transport and
rotating-token/error handling. It still needs a deliberately bounded HTTP
transport/runner and complete hand lifecycle, rather than another decision codec.

Before live play, implement terminal receipt parsing separately from observation
construction: folds, uncalled-chip refunds, all-in runouts, ties, revealed-card
showdowns, auto-muck/disclosure and exact final stacks/net winnings. Replay all
public action prefixes; independently verify every settlement with sufficient
reveals. Report hidden-card accounting limits explicitly. Never synthesize hidden
cards to claim showdown verification. Journal source/model/options/service
identity and hash chains, redact tokens, and preserve failures and truncation
receipts. Ambiguous mutating POST timeout is terminal; no silent retries or
replacement hands. Establish service permissions/rate limits, stable build or
honest unpinned-service identity, seat alternation, token/session recovery and a
maximum request count before an exploratory match.

Recommend a separate **20-hand balanced-seat live-protocol pilot**, subject to
owner approval and a reviewed measured budget. Predeclare translation off, one
hash-pinned audited HU200 average, maximum **1,020 requests** (20 `new_hand` plus
at most 50 actions/hand), one in-flight request, **10-second timeout**, no retries,
**30-minute total cap** including load, journal replay/accounting and closeout.
The timeout worst-case is 10,200s, so reaching the request cap cannot be promised
within 30 minutes; stop at either bound and keep the partial. Network latency and
terminal semantics are unmeasured. The #213 0.675s new-hand connectivity check is
not a per-hand quote. Use this small pilot to measure latency, requests/hand,
model-load/RSS, raw-log growth and failure rates before quoting any larger run.
M1 memory/swap/disk/AC/pressure guards should remain tied to fresh admission.

Only after this qualification should the owner settle acceptance criteria and
approve a fixed exploratory/confirmation budget. The earlier proposal is 10,000
hands per seed, three independent training seeds for confirmation. Public HTTP
server deals are unpaired; balanced seats and fixed nonoverlapping uncertainty
blocks do not create duplicate-deal control. Publish all outcomes and uncertainty,
coverage/fallback, failures and service limits regardless of winnings. The current
single seed and small scripted smoke provide no strength or milestone qualification.
