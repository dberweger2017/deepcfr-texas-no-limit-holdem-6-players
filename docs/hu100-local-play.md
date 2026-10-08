# Explicit HU100 research play

This adds local human and bot-vs-bot play for the fixed #207 terminal average.
It is research access, not v0.5 publication, policy selection or a strength claim.
The released catalog and v0.4.2 default remain unchanged; HU20 sessions and older
journals remain replayable.

The terminal pin comes from #207 head `f0ff3c1ffc4f0002a0c7ab22751d212aa76cdda3`,
`docs/reports/native-hu100-growth-1b-artifacts/model-index.json`: 1,173,264,021 bytes,
SHA256 `47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9`,
checkpoint `cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec`,
seed 2026100601, iteration 885307, 41,010,014 entries, 1,000,002,065 actual nodes.
The loader checks regular-file size/hash, native HU100 header/schema/format,
checkpoint lineage, averaging recipe and uniform zero-mass behavior. It uses
the same compact average reader as the arena. No export is regenerated.

## Retrieval gate and use

As checked October 9, #207 is draft/open and closeout stopped on its original
512 MiB swap-growth guard; its partial ZIP has no accepted archive/readback/cloud
receipt. The owner's later 3 GB swap-cap authorization does not itself accept
that ZIP or complete closeout. This task leaves M4 originals and the partial
ZIP untouched. **Retained-model runtime verification is blocked until #207
closeout permits retrieval.** Fixture qualification does not claim a real-model
smoke check. After acceptance, use the current canonical RESULTS_INDEX entry to
retrieve the named average into a fresh ignored nonsynced `models/` directory,
verify archive/member SHA256 and record the Drive ID, accepted member locator,
hash and exact retrieval command there. Do not use the planned partial-ZIP member.

```sh
python -m src.play_api.server \
  --hu100-research models/retrieved/pr207/average.gz --stack-bb 100 \
  --data-dir results/hu100-play --source-version "$(git rev-parse HEAD)"
```

`--translate-off-menu` explicitly enables #205's unchanged public-history
translation (512 states, 128 events); omission disables it. Use a different data
directory for each setting: session identity and idempotent acknowledgments
reject reusing a session with another inference configuration. Translation keeps
the real wager/menu, records deterministic witness telemetry and uses one sampling
draw. Missing witnesses remain uniform. No implicit 200BB scaling is supported.

Both human modes and the research self-play spectator reset each hand to
10,000 chips/seat, 50/100 blinds, integer chips and chip unit 0.01; button alternates.
HTTP creation does not accept arbitrary stacks, blinds, seeds or policy settings.
`--stack-bb` asserts compatibility rather than overriding the trained game.
Mixed HU20/HU100 spectator policies are rejected. Each session/hand records its
table, model hash and inference identity; net accounting always subtracts the
recorded initial stack. Browser labels use the loaded configuration.

The private journal retains legal observations, action menus/probabilities,
sampling starts, selected actions and deterministic translation receipts for
reproduction. Human responses/history exclude seeds, RNG, hidden bot cards and
private decision receipts. Spectator inspection is intentionally privileged to
the viewer but each bot still receives only its own immutable legal observation.
`--verify-session ID` checks retained hands and in-progress decisions, exact
sampling, legal bounds, observations, receipts and settlement totals. Old human
journals lack decision receipts and receive their original action/event replay.

## Qualification and limits

Small generated average fixtures exercise complete 100BB hands, free off-menu
wagers, translation witnesses, restricted menus, settlement/reset, lost replies,
process restart and spectator decision reproduction. Tamper checks cover policy
settings, tables, observations, probabilities and sampling. Existing released
model pins/defaults and HU20 regression suites are checked as well.

Independent review cleared the legacy resume/identity and browser label findings;
the final fixture suite passes 92 play tests and three JavaScript playback tests.
A local browser fixture completed one human fold hand and one spectator hand;
both retained SQLite journals independently replay and settle at 100BB. The
[verification receipt](reports/hu100-runtime-fixture-verification.json) records
file hashes, the rejected initial localhost navigation and the corrected loopback
URL. These are generated test fixtures, not #207 policy evidence.

Follow-up real-model smoke after retrieval: run 20 human-driven complete hands
(including exact off-menu raises and all-in calls) plus 20 spectator hands in each
translation setting, restart at an in-progress decision, and run both independent
journal audits. Record counts, source/model hashes, peak RSS and swap growth under
the approved 3 GB cap. This is bounded runtime verification, not training or an
external campaign. Keep generated traces/database in ignored results and index
raw evidence under the owning runtime PR before archive closeout.
