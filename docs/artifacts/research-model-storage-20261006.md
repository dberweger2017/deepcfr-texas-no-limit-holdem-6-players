# Research model storage and M1 checkpoint cleanup — October 6, 2026

T3 Code's automatic checkpoints stage the whole workspace. Two unignored forensic model copies totaling 1.47 GB caused repeated 30-second VCS timeouts and abandoned `.git/objects/pack/tmp_pack_*` files. These were local copies created September 20 for analysis of merged PR #89; the checked Git refs contain no commits for either path. They were never release assets.

The owner approved excluding research models from Git and cleaning abandoned packs. Matching local exclusions are applied to project checkouts on M1 and M4; the permanent `.gitignore` covers model binaries, local planning and research bundles. AGENTS.md makes Research-Cloud the canonical location, with retrieval hashes and explicit exact-path approval for release exceptions. The v0.4.0 published model keeps its existing release download.

## Verified canonical models

### c4k.pt

- Former local path: `/Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players/planning/forensics/scratch/c4k.pt`
- Local creation: `2026-09-20T11:44:12.988163+00:00`
- [Canonical archive](https://drive.google.com/file/d/1s-D-48RWoLx1RfO6ES38QRCyfDfU2Ues/view)
- Drive ID: `1s-D-48RWoLx1RfO6ES38QRCyfDfU2Ues`
- Archive SHA256: `11b16adfd66c54efa03ba4981c46ed99f17067e11f1401dc244b55b8a0f99d7a`
- Member: `results/day-current-4k-2026091902/scenario-0-seed-2026091902/pinned/1024/training-1024.pt`
- Member bytes: 688,892,004
- Member SHA256: `634af5aeb746c6d945f23a19c99af83113a0c2973a4af298abe4715c62b4cc87`

### c16k.pt

- Former local path: `/Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players/planning/forensics/scratch/c16k.pt`
- Local creation: `2026-09-20T11:45:28.448319+00:00`
- [Canonical archive](https://drive.google.com/file/d/1d9ynZM5Ssx-hcHQM9VPRQye9GoT6V-1R/view)
- Drive ID: `1d9ynZM5Ssx-hcHQM9VPRQye9GoT6V-1R`
- Archive SHA256: `4bdf55e7294f11295f2ed8dc99822a49fe215b55a321cefa6ff088e09450c06a`
- Member: `results/day-current-16k-2026091902/scenario-0-seed-2026091902/pinned/1024/training-1024.pt`
- Member bytes: 783,479,055
- Member SHA256: `96a96cc9ab9a7769ad676616c44cf0b9fde99a493b71247a52cb2ec9d57eba1f`

Both complete archive hashes match their previously accepted upload receipts. Each model's freshly computed source and archive-member hashes match the embedded archive manifest. Current native Drive status is uploaded, not uploading, with no conflict and the exact document size; current cloud metadata matches the archive ID, name, size and parent. PR #89 is merged. Dependency checks found no tracked references across the registered worktrees or open handles; two historical output JSON files retain the original model filenames as provenance.

## Retrieve only when needed

Run from a project checkout. `Research-Cloud` resolves to the same Drive folder on both Macs; reading a streamed archive downloads it as necessary. If Drive desktop is unavailable, download the exact linked archive above into an ignored local directory and substitute its path. These commands refuse to overwrite existing destination files. Do not load the checkpoint until both archive and member hashes verify.

```sh
set -eu
set -o noclobber
mkdir -p results/retrieved/pr89

archive="$HOME/Local/Research-Cloud/M4-closed-training/results--day-current-4k-2026091902-20261004.tar.gz"
printf '%s  %s\n' '11b16adfd66c54efa03ba4981c46ed99f17067e11f1401dc244b55b8a0f99d7a' "$archive" | shasum -a 256 -c -
tar -xOf "$archive" 'results/day-current-4k-2026091902/scenario-0-seed-2026091902/pinned/1024/training-1024.pt' > results/retrieved/pr89/c4k.pt
printf '%s  %s\n' '634af5aeb746c6d945f23a19c99af83113a0c2973a4af298abe4715c62b4cc87' 'results/retrieved/pr89/c4k.pt' | shasum -a 256 -c -

archive="$HOME/Local/Research-Cloud/M4-closed-training/results--day-current-16k-2026091902-20261004.tar.gz"
printf '%s  %s\n' '4bdf55e7294f11295f2ed8dc99822a49fe215b55a321cefa6ff088e09450c06a' "$archive" | shasum -a 256 -c -
tar -xOf "$archive" 'results/day-current-16k-2026091902/scenario-0-seed-2026091902/pinned/1024/training-1024.pt' > results/retrieved/pr89/c16k.pt
printf '%s  %s\n' '96a96cc9ab9a7769ad676616c44cf0b9fde99a493b71247a52cb2ec9d57eba1f' 'results/retrieved/pr89/c16k.pt' | shasum -a 256 -c -
```

For an exact historical replay, retrieve the selected member into its former `planning/forensics/scratch/` path instead; that directory is also ignored. The full model receipt records original paths and restoration commands. Treat downloaded evidence as read-only and review current PR status, active dependencies and upload acceptance before later cleanup.

## Cleanup and validation

- Two duplicate local models removed: 1,472,371,059 bytes.
- 82 unchanged, unopened Git-reported garbage packs older than two minutes removed: 33,710,801,880 bytes.
- Model hashes and archive/member manifests matched before model removal. Synced archives, active PR roots and valid Git packs/refs/indexes were retained.
- Full Git integrity and reachable-object checks passed before and after temporary-pack cleanup.
- Equivalent checkpoint staging using a disposable index completed in 1.4 seconds after exclusion; all 13 ignore-rule cases passed.
- M1 measured about 51 GB free after cleanup, with zero temporary packs. Physical free space also changes with Drive caching and active runs.

[Full model cleanup receipt](https://drive.google.com/file/d/1XgRadPHdyCnp17SUg-9t7zDIZa5P-TjB/view) · [Full temporary-pack cleanup receipt](https://drive.google.com/file/d/11pRy40cfqt38Q781JyLO1rAsFoc_yHaa/view)

The local full receipts are in `~/Local/storage-cleanup-receipts/`. No recurring cleanup or forced cloud-cache eviction was configured.
