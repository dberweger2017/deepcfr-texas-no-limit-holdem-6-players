# M1 inactive worktree cleanup — October 9, 2026

Owner-requested cleanup retired **57 worktrees** unused for at least 24 hours: 45 clean source/build checkouts and 12 merged-PR evidence roots. Separate removal batches measured **17.428 GB reclaimed**; **30.323 GB free** immediately after cleanup. Concurrent research and newly created backups affect free space between batches, so these are measured removal-batch gains, not an estimate from directory sizes.

[Complete path-by-path receipt](https://drive.google.com/file/d/1J1OsjpghvHUmhxSEnmdkpnzqChKSCNmb/view?usp=drivesdk) — 7,561,003 bytes, SHA256 `edf33765bf1ac80faec1da8419bfbca79dca5c642078aa935c431b3e94d41379`. [Compact source/archive locators](m1-worktree-cleanup-20261009.json) are checked into Git. The complete receipt records every removed research path, original stat, archive/member/hash, excluded ephemeral file and restore command.

## Preservation and acceptance

- Fresh owning PR status was merged before archival and removal. Open #218, active/recent threads and processes, dirty/untracked work, changed files, active inputs and shared Git were protected. The newly active v0.5 readiness workspace was included in the dependency review.
- Branches were retained and every removed source HEAD is pinned by `refs/cleanup/m1-worktrees-20261009/<full-commit>`. No history was rewritten.
- All 12 new archives have current native uploaded/not-uploading/no-conflict/not-trashed status and independent cloud ID/name/size/parent readback. Private archives have owner-only permissions. New archive members passed local size/SHA256 readback; Google Drive confirmed uploads are trusted, with no remote download or repeated old archive/member hash audit.
- Existing manifests and unchanged original paths, sizes, identities and modification history cover 2,934 PR190 members and 12 PR202 members. New supplements retain their late records and exact prior archive/member locators, avoiding duplicate primary evidence uploads.
- Spectator and Luna-versus-Shield journals, screenshots, audits, failures, model copies and retrieval provenance are archived. Expired local test access tokens, stopped isolated browser profiles and reinstallable browser dependencies were excluded. The original Luna agent rollout outside the worktree was untouched; its unfiltered reasoning is not published.
- No synced file was deleted or forced offloaded. Only the explicitly recorded nonsynced worktrees and temporary archive copies were removed. Canonical cloud archives remain.

Two inactive roots remain because maintained source/configuration references them: `~/.codex/worktrees/blueprint-averaged-extraction/deepcfr-texas-no-limit-holdem-6-players` and `~/Local/native-recovery-hu100-20261008`. Other recent, active or dirty roots were not eligible.

## Accepted retirement snapshots

Every archive embeds `ARCHIVE-MANIFEST.json`. The manifest preserves original paths, sizes, SHA256s and source revisions, maps duplicate file aliases, and identifies previous archive members and excluded ephemeral files. Names below are stored in their designated PR folders; the compact receipt gives folder IDs, native paths and manifest SHA256s.

| Owning PR | Archive | Bytes | Archive SHA256 |
|---|---|---:|---|
| #109 | [m1-retirement-pr109-20261009.zip](https://drive.google.com/file/d/1hFCVwlPxq0NZea7jMvjRY_nkEWt81cYB/view?usp=drivesdk) | 1,500 | `4d73596dfca0c5b6d2e24e08b5446d1db1eb39591be95faffb73bd6d81c5acac` |
| #112 | [m1-retirement-pr112-20261009.zip](https://drive.google.com/file/d/1Tag_3_GXhEf_xMZ4RpxPe_26UPd8HjTr/view?usp=drivesdk) | 17,474,323 | `c0b052ac809e6a1872939b655248335b84e4a02e3b62856a896362a03815ecf4` |
| #200 | [m1-retirement-pr200-20261009.zip](https://drive.google.com/file/d/1wVRs9F7EmjxVyOyQ414CsU-v7tNAxPd2/view?usp=drivesdk) | 649,870 | `20ef52888581f76e1d88c5bbd4a53913284c76b364965e313dda6ca531614745` |
| #105 | [m1-retirement-pr105-20261009.zip](https://drive.google.com/file/d/1jwgbMXtgPj4259mEsEygDBqSGce7IfVL/view?usp=drivesdk) | 1,544 | `6f9cf5375ac6a6e873f074b1d4e59d751edc15cf35ffb444f861a859eaf7b14b` |
| #190 | [m1-retirement-pr190-20261009.zip](https://drive.google.com/file/d/1yPKWhlg18LNS03Ykm3oxs9NlKUyct2U2/view?usp=drivesdk) | 843,760 | `8262edac135914c9739ea7eaf10c67795a53739f00b9a695fadcea9803884bf9` |
| #199 | [m1-retirement-pr199-20261009.zip](https://drive.google.com/file/d/1mpBNQLGeOWqeUjBRRdPWzjDB2sHiW51A/view?usp=drivesdk) | 253,345,864 | `9e501b9882c0a00777e6fcabf53c7611105553f0a6093e6dc5698c03862826f4` |
| #193 | [m1-retirement-pr193-20261009.zip](https://drive.google.com/file/d/1a31dazuAS1mzvbj9sG4JXO83gs2cBpBW/view?usp=drivesdk) | 188,162,257 | `139c73833112a6c936d46748fab060b1bc4e40c721f6a765d3bd0ed88d43c746` |
| #201 | [m1-retirement-pr201-20261009.zip](https://drive.google.com/file/d/1rB70ruysjH9jpz5ixO5gO1W_v_0eqTZd/view?usp=drivesdk) | 11,526,238 | `38996ba5e081cbbe20f691a6178283e172d9fcc174075612417b9689a855fccb` |
| #202 | [m1-retirement-pr202-20261009.zip](https://drive.google.com/file/d/1HO96dGJ9Zcf49ftsR8pAVKgYIpEVVzB2/view?usp=drivesdk) | 158,035 | `03c41fc3d9175f04cf72d72f4fd9cc387da1d5b3bae4c01931f3c1a5fabd2c76` |
| #174 | [m1-retirement-pr174-20261009.zip](https://drive.google.com/file/d/1VMmc8AsL1Aku17oJtTcGkmjRExmjcNCM/view?usp=drivesdk) | 144,836,983 | `b27bd27eecddec1ef5296a77ab2d2fac83224bfdeffe19947bad4736ffcad9cc` |
| #166 | [m1-retirement-pr166-20261009.zip](https://drive.google.com/file/d/1OMPMLY4o8pNjjcZUgocsUuq7pLttMxQf/view?usp=drivesdk) | 429,372 | `7d208f0c8a758bef402e480af6d91bf121dfbb7abdcf6e7db1ad7b7d9ca98f4e` |
| #188 | [m1-retirement-pr188-20261009.zip](https://drive.google.com/file/d/1LpUahSjRwWNDFcZMdDSnQvWoMJxi5Nis/view?usp=drivesdk) | 2,838,542 | `c34d29e3421a614eaa581b435a83e85e83e6ccc2ed1868214468aad5233028b1` |

## Model restoration locators

Use the owning PR archive above. The PR199 publication copy maps to its identical archived download member; both original paths are recorded.

| PR | Original relative path | Archive member | Bytes | Member SHA256 |
|---|---|---|---:|---|
| #174 | `models/O-2026100601.average.jsonl.gz` | `models/O-2026100601.average.jsonl.gz` | 143,429,027 | `a5e9d0fc6f4a448640f52f508187f779e43a0a8fd41158207f03de82adc219e1` |
| #193 | `models/B100M-HU20-current-seed-2026093001.json.gz` | `models/B100M-HU20-current-seed-2026093001.json.gz` | 40,144,034 | `4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf` |
| #193 | `models/O1B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz` | `models/O1B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz` | 142,677,367 | `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d` |
| #199 | `results/v042-release/draft-download-01/O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz` | `results/v042-release/draft-download-01/O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz` | 249,237,403 | `15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae` |
| #199 | `results/v042-release/publication/O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz` | `results/v042-release/draft-download-01/O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz` | 249,237,403 | `15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae` |

## Restore

Restore source with the exact `restore_command` in the compact receipt. It creates a detached worktree at the preserved source revision. Retrieve the listed archive into a **fresh nonsynced ignored directory**, verify its recorded whole SHA256 before research use, then read `ARCHIVE-MANIFEST.json` and extract the named member. Verify the selected member SHA256 before using a model or input. For aliases, extract `member` and place it at its recorded original relative `path`. For `previous_archive_references`, retrieve the stated archive ID and extract its stated member. Never overwrite active inputs.

The full receipt includes original paths and excludes expired authentication/browser state from restoration. The PR174 model is self-contained in its new private archive; restoration does not depend on the historically referenced PR171 local path, which is currently absent and was outside this selection.

After downloading the selected Drive archive to a fresh ignored path, these commands show the manual extraction flow (replace the archive and member with the receipt's exact values):

```sh
shasum -a 256 results/retrieved/selected-archive.zip
unzip -n results/retrieved/selected-archive.zip 'ARCHIVE-MANIFEST.json' 'listed/member/path' -d results/restored-worktree
shasum -a 256 results/restored-worktree/listed/member/path
```

Compare both printed hashes with the pinned receipt/manifest before research use. Existing archive references have their own archive SHA256 and named member in the new manifest.

## Validation and limitations

Selection was checked against file/Git activity over 24 hours, current PR states, recent agent sessions, active T3 bindings, open process handles, shell references, source/configuration paths and symlinks. Immediately before removal, root/HEAD/ignored-file inventory and every original size/inode/device/mtime/ctime were rechecked. Non-forced `git worktree remove` succeeded, each removed root was absent, and the primary checkout retained its source revision and pre-existing untracked files. Old PR190/PR202 manifests were reused without rehashing their payloads.

The connector rejected the 253 MB PR199 upload before invocation because its input limit is 100 MiB; the three larger snapshots were uploaded with the existing native Drive client and then accepted independently. An initial spectator preflight followed dangling disposable Chrome-profile symlinks and retained the root; a corrected `lstat` check established unchanged identity before the separate removal pass. Both records are in the full receipt.

This is a dated cleanup record. No research run, model change, release, unattended cleanup or recurring task was started.
