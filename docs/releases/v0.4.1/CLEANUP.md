# Release cleanup review

#180 is merged at `3b742ad667f640b1b017bf692b8af1a8734aefc8`. Its research-model/bundle ignore rules and verified retrieval policy are preserved. No model enters this release's Git tree: inference assets remain outside Git, staged for the eventual GitHub release.

A tracked-file inventory found **no AppleDouble `._*`, `.DS_Store`, scratch/planning directories or training/model blobs** to remove. Add correctly cased macOS metadata ignore patterns so these cannot enter future checkpoints. No history rewrite or force push, and no local research-original cleanup is performed by this release PR.

The four tracked files over 10 MB are useful, referenced research evidence. Retain them; deleting them requires a separate owner OK and verification of their archived restoration dependencies:

| Item | Bytes | Decision and reason |
| --- | ---: | --- |
| `docs/reports/hu20-exact-turn-check-artifacts/main03-overfold-groups.jsonl.gz` | 13,254,365 | Retain: linked by the exact-turn report and consumed by `audit_board_pooling_keys.py`. |
| `docs/reports/hu20-exact-turn-check-artifacts/main03-results.jsonl.gz` | 14,871,344 | Retain: report links the full atomic results; inventory pins their hash. |
| `docs/reports/hu20-stackoff-artifacts/decisions.csv.gz` | 11,445,573 | Retain: raw decisions used by stackoff checks, reports and evidence manifests. |
| `docs/reports/hu20-stackoff-artifacts/generated-hands.jsonl.gz` | 10,531,128 | Retain: native replay records used by stackoff checks, reports and evidence manifests. |

No clearly dead script is identified: uninvoked historical research scripts are not presumed junk while their reports and restoration workflows remain. No uncertain item is removed. #166's open work roots/dependencies and all synced Google Drive files are untouched.

**Removal list: empty.** Prevention-only changes and this review are a separate cleanup commit. All research and verification originals remain; the new member-hashed release archive adds evidence without overwriting, deleting or evicting existing archives.
