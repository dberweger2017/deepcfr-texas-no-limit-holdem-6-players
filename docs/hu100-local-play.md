# Explicit HU100 research play

This adds local human and bot-vs-bot play for the fixed #207 terminal average.
It is research access, not v0.5 publication, policy selection or a strength claim.
The released catalog and v0.4.2 default remain unchanged; HU20 sessions and older
journals remain replayable.

The terminal pin comes from #207's model index, unchanged at its reviewed closeout
head `fbe19c17f0fbd474e5ea26393cc16c9ddeafaf27`,
`docs/reports/native-hu100-growth-1b-artifacts/model-index.json`: 1,173,264,021 bytes,
SHA256 `47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9`,
checkpoint `cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec`,
seed 2026100601, iteration 885307, 41,010,014 entries, 1,000,002,065 actual nodes.
The loader checks regular-file size/hash, native HU100 header/schema/format,
checkpoint lineage, averaging recipe and uniform zero-mass behavior. It uses
the same compact average reader as the arena. No export is regenerated.

## Retrieval gate and use

On October 9 the owner corrected the earlier closeout status: #207's repaired
archive and independent review are complete; merge awaits CI. Its new retry ZIP
is 20,517,119,304 bytes, SHA256
`ba3e82d8fa79be32d445c54eb240069c4a717cb75af9713f3eaf7ac86364fddf`.
Connected Drive confirms [the archive](https://drive.google.com/file/d/1iowJoQBQqB3tLRnU6GD0JcniF0qDIvgj/view)
name/size and parent `1qhlOHmphBGSFfiM82S7T4B_KhdabyRUS`; current native Foundation
status reports uploaded, no pending upload/error and non-nil zero unresolved
conflicts. The terminal average was streamed read-only from the existing fully
allocated M4 Research-Cloud ZIP into ignored M1 `models/retrieved/pr207-20261009/`.
The whole ZIP, embedded manifest and selected member all passed SHA256 checks.
No second archive copy was made, no remote cloud bytes were separately downloaded,
and no original, partial ZIP or other campaign input was changed. The original
guard/cleanup failures remain historical evidence; the failed partial ZIP remains
an invalid restoration source. [RESULTS_INDEX](../RESULTS_INDEX.md) records the
exact member, hashes, Drive IDs and retrieval command. For future retrieval use
that accepted archive and a fresh ignored nonsynced directory; never overwrite
an active model.

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

Retained-model qualification now passes **80 complete 100BB hands**: 20 scripted
human-seat hands and 20 self-play spectator hands per translation setting, through
the actual loopback HTTP handler. Four new worker processes per setting recovered
durable journals; two deliberately repeated advance requests returned identical
acknowledgments across different process IDs. Both settings exercised 550-chip
off-menu raises and all-in calls somewhere; translation-disabled human play did
not contain an all-in call, while its spectator supplied that coverage. Eight
translation witness receipts were retained. Incompatible table/model selection
and changed translation identity were rejected. Independent journal modules
regenerated policy decisions and settlements; separate review replayed all hands,
legal observations/actions, sampling continuity and event/accounting hashes,
and inspected all 593 HTTP responses for expected rejections and privacy.

[Retained verification receipt](reports/hu100-runtime-retained-verification.json)
records the exact model/source/helper hashes, counts, raw-file retention and
resources: load 230.97 seconds, sampled summed family RSS 2.554 GB, maximum total
swap 1.606 GB, minimum free disk 34.696 GB; no guard breach. The disk floor was
15.5 GiB, summed RSS ceiling 8 GiB, total swap ceiling 3,000,000,000 bytes and wall
cap 1,800 seconds. Loading benefited from retrieval/hash-warmed filesystem cache;
HTTP workers inherited one immutable validated compact model rather than loading
another copy. Summed fork RSS can count shared pages twice. Separate cold CLI
starts are not measured by this receipt. A missing monitoring dependency and a
pre-hand helper session-ID mistake were repaired and preserved; no retained hand
failed. The run and all workers are stopped. No training, external campaign or
strength inference follows from this runtime check. Keep the ignored model and
raw evidence while PR212 remains open; archive closeout is pending.
