# External heads-up benchmark readiness

**Recommendation: keep 100BB and seek a confirmed 100BB Slumbot endpoint or a
maintainer-supplied frozen 100BB opponent before evaluating.** Slumbot is the
strongest practical first access candidate among the interfaces checked, but its
public API is **200BB**. This PR prepares public protocol boundaries and refuses
the mismatch before a policy or transport call. It does not redefine HU100,
scale stacks, train another model, select a release or run an external campaign.
The opponent and acceptance rule remain owner decisions.

## Verified access and candidates, October 9, 2026

Primary sources are linked below. Availability means exactly what was checked;
source code, trained weights, live service access and matching rules are distinct.

| Opponent | Access and protocol | Compatibility and resources | Limitation/decision |
|---|---|---|---|
| **Slumbot public service** | Official site and downloadable [sample](https://slumbot.com/sample_api.py) reachable. Anonymous `POST /slumbot/api/new_hand` returned HTTP 200 in 0.675s, seat 0, opening `b200`, two own cards, empty board and a token. One request, zero policy actions; [receipt](reports/slumbot-access-20261009.json). HTTPS JSON `new_hand`, `act`, optional `login`; rotating token. | Official API page explicitly fixes 50/100 blinds, reset 20,000-chip **200BB** stacks. Street-local bet-to `bN`, `k/c/f`, slash street boundaries. First hand is BB; use even hand counts. No paid access used. Remote inference avoids opponent RAM on M1; rate limits, service version pin and campaign permission are unconfirmed. | **Incompatible with fixed HU100.** No advertised stack override. Request a genuine 100BB endpoint/build through an owner-approved contact step, or keep blocked. 0.675s is connectivity latency, not a hand throughput quote. |
| **Slumbot2019/2017 local source** | [Author's 2019 repository](https://github.com/ericgjackson/slumbot2019), inspected `a74c99dd5e6e2fb50118e3990a25210e9dd0f6b3`, supports CFR, solving and head-to-head tools; MIT source. GitHub lists no releases. | Configurable game/betting abstractions do not supply a trained 100BB champion. Author tutorial builds buckets and trains a policy. Training/storage quote on M1 is not measured. | Source is available; matching established trained opponent bytes are not verified. Retraining or changing a 200BB opponent's game would need a new declared scope and opponent identity. |
| **DecisionHoldem** | [Author repository](https://github.com/AI-Decision/DecisionHoldem), `a9ea9a545c7bb24f4e657bc6d1f75af66aa1bb51`; C++ source and shared-object interfaces, AGPL-3.0. README requires six external data files from Baidu Netdisk. Source/API GET verified, required weights not downloaded or accepted. | Slumbot bridge hardcodes 20,000 chips. README's training used 48 CPU cores for 3–4 days. Existing binary architecture, matching weights, supported 100BB policy and inference RAM/time are unqualified. | No usable matching 100BB opponent acquired. A changed configuration does not validate old weights. No download of bulky assets or training authorized here. |
| **DeepStack / Libratus** | [Original DeepStack team's public example](https://github.com/lifrordi/DeepStack-Leduc) is explicitly **Leduc**, requiring Lua/Torch; not a released full NLHE service/policy. [Libratus paper](https://www.science.org/doi/10.1126/science.aao1733) establishes the research opponent, not downloadable evaluation access. | No verified compatible full HU100 executable, weights or live bot API in inspected primary material. Reproductions are different opponents and require their own qualification. | Cannot claim actual access to the research systems from paper/code availability. |
| **ACPC / OpenStackTwo** | ACPC is a match protocol/server, not an opponent. DecisionHoldem README links the [OpenStackTwo service](http://holdem.ia.ac.cn/#/battle) and its historical client. | No current operator-authorized, rule-pinned 100BB endpoint or trained artifact verified. ACPC bots frequently use different fixed stack contracts; a match server's editable game file does not make a bot compatible. | Useful future adapter target only after a concrete opponent and actual access are established. No credentials, paid account or speculative campaign. |

[Slumbot official site/API tab](https://slumbot.com/) was inspected in the live
browser; official sample SHA256
`17ab1f6c1a25db1822cef7f34ed67b6516ac65ea24897adcc92d8f2db8f71505`.
The downloaded official sample was read for wire semantics, not copied into the
repository. The one connectivity response is summarized without token/private
cards. It does not verify terminal accounting, bot strength or endpoint stability.

## Implemented boundaries and exact limits

`src/arena/external/interface.py` declares the table contract, observation adapter
and injected JSON transport. Admission compares native trained game/schema,
initial stacks, blinds, players, rake/ante, per-hand reset and amount convention;
it rejects Slumbot's 200BB before network access. No live campaign runner or HTTP
transport is shipped. `SlumbotConnection` only exercises wire/token handling via
injected transports, and cannot currently admit any released/research policy to
Slumbot. Future matching policies require a separately reviewed contract.

The Slumbot codec validates response grammar, exact chips, own/board card counts,
duplicate cards, action order, legal sizes, street transitions (including the official optional final separator) and client turn.
It reconstructs **decision prefixes** from public betting events using the
existing public reducer and observation replay: no simulator, invented deck,
opponent hand, seed or payoff enters inference. Terminal/error/non-client
responses are rejected. Token, extra evaluator fields and opponent-card additions
are dropped by an allowlist. Terminal all-in `b20000c///` is covered as a refusal
fixture; it cannot accidentally become a decision. Live response token rotation
is handled at the transport boundary; responses without an update retain the current token.
Official `error_msg` responses stop the connection, including empty error messages. Mutating POSTs have no documented
idempotency key, so ambiguous timeouts are failures, never blind retries.

Protocol fixtures compare public prefixes with the independent pinned native
engine, including preflop seat mapping, postflop order and street-local bet-to
versus lifetime commitment. `replay_record` retains filtered public input,
reconstructed observation and selected action. `PrefixJournal` creates a new file,
hash-chains records and fixed failure classifications, and verifies every retained
prefix; an externally recorded tail hash detects truncation, and empty journals
are rejected. Store raw journals under ignored `results/`, archive/index with owning-PR
provenance per artifact policy. Model/source/options identities belong in the
journal header; tokens remain owned by transport.

**This is not completed full-hand external integration.** The service's exact
terminal disclosures, all-in runout conventions, net winnings/refunds, bot version
pinning, rate limits and timeout recovery still need a bounded matched-game
qualification. Prefix replay alone cannot independently verify hidden-card
showdown winnings. Terminal evaluator receipts must remain separate from policy
inputs; compare raw chip sums to server net winnings and verify every settlement
whose revealed cards permit it. Do not invent hidden cards to claim full replay.
A future ACPC adapter must convert its **total commitment** raise convention
separately; never reuse Slumbot's street-local codec blindly.

## Proposed frozen evaluation, requiring owner approval

Choose **Slumbot at verified 100BB**, if its maintainer supplies that exact endpoint
or an established pinned policy. Keep this as the preferred access route; do not
claim it exists today. Alternative: the owner may approve a **separate HU200
research task** with a correctly trained/qualified 200BB model and its own
strength definition. That would not satisfy the currently declared HU100 target.
No paid access, bulk download, training or game-scope extension is approved here.
No message has been sent to any maintainer.

After compatible access and a reviewed operator agreement, propose **5,000 fresh
swapped-seat deal blocks = 10,000 completed hands per fixed policy seed**, **three
independent training seeds** for milestone confirmation. The present #207 lineage
alone can supply development evidence, not the roadmap's multi-seed v0.5 gate.
Freeze model SHA256, source/adapter version, translation setting (off primary;
no post-hoc selection), opponent artifact/service build, rules, independent root
schedule, exact sample count and stopping rule before play. No choosing the best
checkpoint, extending weak results or pooling prior deals.

If access is HTTP-only and does not permit controlling deals, use **10,000 hands
per seed with balanced alternating seats**, explicitly **unpaired server deals**;
report this weaker design and service sequence/session effects. Predeclare 100
nonoverlapping 100-hand blocks for batch uncertainty, inspect serial dependence
and disclose that block independence is an approximation. Do not claim duplicate
deals, deterministic opponent reproduction or exact service-version stability
without support. For controlled offline duplicate deals, Student-t uncertainty
over independent block means is the primary estimate, in BB/100 and raw chips.
Report each seed and cross-seed variation separately.

Proposed acceptance is **credible evaluation readiness**, not beating Slumbot:
all scheduled hands complete with balanced seats; zero information leaks,
illegal actions, protocol/accounting mismatches or silently retried hands; all
policy decisions/prefixes reproduce; terminal accounting limitations are disclosed
and every verifiable settlement passes. Missing external replay/disclosure or
opponent identity guarantees are explicit qualification blockers, not waived
checks. Publish fixed-sample winrate, uncertainty, coverage/translation/fallback
rates, failure counts and costs regardless of result. Positive profit is not a
required gate unless the owner explicitly adopts one. A precision miss is reported
as inconclusive without extending the sample.

Before any large evaluation, run a separately approved 20-hand protocol/timing
pilot at compatible stacks, count actual decisions/requests and measured M1 model
RSS/load time, opponent latency and raw-log growth. Quote wall time with 2x measured
headroom, disk floor, memory and the owner's **3 GB swap cap**, explicit maximum
requests and timeout policy. The single connectivity request is insufficient to
quote 10,000-hand cost. Dollars spent so far: zero. A public no-fee API is not
permission for an unlimited automated load.

**Remaining blockers:** no established 100BB opponent interface or artifact is
confirmed; full-hand terminal/refund/disclosure and service-version qualification
are unfinished; #207 closeout has not yet permitted retained-model retrieval;
only one HU100 training seed exists; owner has not approved the opponent, fixed
campaign quote or proposed acceptance criteria. This PR leaves concrete choices
and testable boundaries ready for review, unmerged.

## Local qualification

Independent review found official `error_msg` and optional-token-update mismatches,
an omitted final street-separator case and empty-journal acceptance; all are fixed
with deterministic regressions. Reviewer ran 58 focused tests and 790 public
decision prefixes from 200 simulated 200BB hands against the pinned native engine:
player commitments/stacks, pots, streets and legal bounds matched. These are
isolated protocol fixtures, not an external match or a 100BB strength result.
No open independent-review finding remains. Staged artifact and whitespace checks
must pass at commit; final-head CI remains a separate merge requirement.
