# Windowed K1 blueprint extraction protocol

This dependent experiment replays the three completed K1 continuations in
draft PR #110 from the byte-pinned 5.83M parent. Each replay must reproduce its
original final checkpoint byte for byte before a playing comparison begins.
The trainer, action abstraction, regret updates, and trainer random streams
are unchanged. Extraction uses separate deterministic random streams.

The fixed capture window contains eight equally weighted published profiles
at the first completed outer iterations crossing the node milestones in
`configs/blueprint/windowed-extraction-m4.json`. The earlier ten million
additional nodes are warm-up. A preflop UPDATE-STRATEGY collector samples the
target seat's actions, enumerates opponents' abstract actions, and samples
chance through the native engine. Its counts accumulate separately from
training. Because abstract keys can merge underlying histories, a counter
aggregates all target visits bearing that key; it is an own-reach-weighted
sample estimate for this abstraction, not a perfect-recall result. When a
preflop key has zero collected mass, play falls back to the final current
policy and records that event.

Postflop policy for each key is the arithmetic mean of eight current
regret-matched action distributions. A missing snapshot contributes uniform
probability for that key's abstract menu. Snapshot files are sorted streams;
the index merges them without loading eight training tables. Each indexed
artifact has its own SHA-256 identity and a manifest recording source hashes,
capture points, collector parameters, key schema and fallback rules.

All C/P/F/A variants use the same verified index and differ only by selected
distribution. The no-free-fold rule is applied after lookup and extraction.
The historical parent and uniform controls use the existing canonical-safe
lookup. The scripted and random schedules are coupled across all arms, with
fresh seeds. The two scripted primary contrasts are A minus C, averaged
within each block across the three continuations, and A minus U_safe;
Bonferroni 97.5% block intervals apply. P/F are exploratory diagnostics.

Before opening playing outcomes, run a resource-only capture and collector
preflight on one existing final K1 checkpoint, then freeze collector roots and
evaluation counts. The campaign has a ten-hour wall ceiling, 10.5-GiB process
RSS guard, and disk guard. A failure retains partial artifacts and ends that
phase without silently changing the frozen comparison.

The M4 resource-only preflight on the existing `2026092701` K1 checkpoint
loaded in 30.08 seconds. A full-size sorted snapshot took 25.10 seconds and
247,911,650 bytes. Sixteen collector roots per seat took 24.26 seconds,
visited 372,932 states and produced 69,274 action counts at 65,496 keys.
Peak RSS was 7,102,070,784 bytes; system swap remained at 761.38 MiB and
reported free memory was 69%. These measurements fix sixteen roots per seat
per capture and the 4,096 scripted/1,024 random block schedules in the plan.
No new playing outcomes were examined during this preflight. The retained
preflight directory is `results/windowed-preflight-16-20260927` in the M4
PR #111 checkout.

This protocol tests readout from the same trained strategies. It cannot
establish a general six-player equilibrium guarantee or promote a player.
