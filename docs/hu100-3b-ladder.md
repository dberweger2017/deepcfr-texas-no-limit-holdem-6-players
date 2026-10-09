# HU100 3B nodes and direct ladder: prospective protocol

Owner-authorized overnight on the free 10-core/16-GiB Apple M4, after the
independent K50 bench. Fresh current-main isolated checkout; no trainer edits.
Seed 2026100601, `train --stack-bb 100 --roots-per-seat 1 --average-rule
opponent-sampled`, unchanged linear CFR, v1 cards/history/menu and uniform
missing/zero fallback. Fresh training; no resumed archive. Milestones 500M,
1B, 2B, 3B stop at the first complete iteration. Telemetry throughout.

Before advancing beyond each gate, checkpoint SHA256 must equal #207:
500M `35f4b46e6c490573b2cf3ebe4953c269a7793845f425dd53e296fc64be439788`;
1B `cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec`.
The first fresh segment uses #207's 57,658,644 entry ceiling through 1B
so the checkpoint header bytes match too; after both gates, continue that
fresh local state with the newly forecast soft-limit ceiling. No archived
state is resumed. Stream-export current/average and fully audit every evaluated checkpoint.
Exactness mismatch stops without scientific retry. Existing hash-verified
500M/1B originals are indexed and excluded from duplicate archive copies.

At #207's 1B, 41.01M entries use 3.56 GiB, about 93 B/entry. Falling growth
exponent ~0.45 forecasts 55M at 2B and 65M at 3B. Conservative capacity is
floor((7 GiB−100 MB)/110 B), **67419934 entries** (calculate from constants
at execution). Soft/hard whole-family RSS 7/9 GiB, normal pressure, >=15%
system free, **3,000,000,000 bytes total swap**, >=16 GiB free disk and AC.
Soft stop/entry cap saves at a complete iteration; terminal below 3B is a
valid result. Guard breach, information leak, invalid play or accounting
failure stops this stage and is reported on the PR. No trainer fixes or retries.

Direct matches reset both stacks to 100 BB every hand, blinds 50/100, no rake.
Every fresh deal plays twice with seats swapped, independent duplicate blocks,
private action streams and both policies confined to menu actions. Translation
off. Primary terminal (3B if reached) versus 1B; descriptive 2B versus 1B and
1B versus 500M. Same frozen direct root 2026100922202, scenario per rung;
timing-only pilot root 2026100922201. Check physical deal disjointness against
all earlier HU100 schedules, including #215. Fully replay/reproduce every
final hand, action probability, key, menu and settlement.

Freeze direct sample from the timing-only pilot before final hands. Target
ordinary two-sided 95% Student-t interval half-width about 5 BB/100. Prior
HU20 direct SD was approximately 333 BB/100 per duplicate block; scaling by
stack ratio five gives a conservative planning SD 1,665. Thus ceil((1.96 *
1665 / 5)^2)=426,006, rounded to **524,288 blocks/rung** when measured costs
fit the overnight quote. Actual half-width is reported; no outcome-based
extension. If costs require a smaller sample, freeze the largest power of two
admitted by disk and whole-family memory, report the projected precision
limitation prospectively, and retain the terminal checkpoint unchanged. The ten-hour
threshold reduces scope: it never cancels the required primary. Its full measured
time quote, including replay/reproduction, becomes its own frozen deadline.
Improvement requires the primary interval lower bound >0. Other rungs are
nominal descriptive 95%; a three-comparison Bonferroni interval may be reported
as context, without changing the sole primary decision or selecting a rung.

Secondary: terminal and 1B against all five unchanged scripted opponents,
translation on with exact default 512-state/128-event settings, 4,096 blocks
per opponent. Common paired deals, fresh root 2026100922204 (pilot2203),
private streams, coverage and reached visit bands. These are descriptive;
no additional primary pass opportunity. If combined pilot quotes for A+B
exceed about ten hours, complete A and B training/direct primary, skip B's
scripted secondary (and admit descriptive rungs only if time/storage fit).

Post measured train/save/export/audit/direct/secondary time and storage quote
on this PR before final play. One independent source review before final run;
one end evidence review. ZIP research in `~/Local/Research-Cloud/PR-<n>-hu100-3b-ladder/`
with member hashes, failures and provenance; index required restoration.
No models in Git, cleanup of other PR files, paid compute, tag, publication
or release; v0.5.0 remains the owner's decision. Merge on green checks and
no open findings; update roadmap on landing.
