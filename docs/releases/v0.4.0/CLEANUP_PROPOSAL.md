# v0.4 remote cleanup proposal — review snapshot

Snapshot: 2026-09-30 UTC, GitHub REST branch/release pages (all 48 branches and all 6 releases), `gh pr list --state all --limit 500` (108 PRs), `git ls-remote --tags`, M1 and M4 worktree inventories. This is an **exact-name proposal, not authorization to delete**. Refresh every state and SHA immediately before any approved operation. No remote branch, tag, release or asset was deleted.

## Branches

KEEP includes every open PR head/base, checked-out worktree, unmerged or unexplained ref, and branch-hosted PR-body reference. A DELETE proposal applies only where the remote head equals the merged PR’s final head and no known live dependency remains. Squash merges do not imply ancestry; the PR record, head equality and verified backup are required. External unindexed references or automation may still require changing a proposed DELETE to KEEP.

| Remote branch | Head SHA | PR/merge evidence | Known dependency | Proposal | Recovery ref after backup |
| --- | --- | --- | --- | --- | --- |
| `feature/20bb-robustness` | `848e958155b8fd91a6906f7f49fffa09041436f5` | #114 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/20bb-robustness` |
| `feature/benchmark-opponents` | `bda25895e33d3255785df7b78110fcecd8f45b4c` | #45 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/benchmark-opponents` |
| `feature/blueprint-averaged-extraction` | `349a6cb167ec5e1850b6300aa6a079f94803978e` | #111 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/blueprint-averaged-extraction` |
| `feature/blueprint-history-coverage` | `c509c1b400cd83f6e24a2257099d686e908fbc8a` | #98 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/blueprint-history-coverage` |
| `feature/blueprint-local-cfr` | `3a055cf84d640ba41e2276544150437ba4c4787a` | #106 merged (head match) | PR body #106 | **KEEP** | Keep remote ref |
| `feature/blueprint-mixed-table-demo` | `7d9fe4db594f0ad9ea7cb843342d7e5b82fe0fe2` | #101 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/blueprint-mixed-table-demo` |
| `feature/blueprint-postflop-search` | `7adde48ffeb5acac5278a0157628fd5902ab1a02` | #103 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/blueprint-postflop-search` |
| `feature/blueprint-runpod-learning-check` | `5a3964b8e3ba7af0b5dc32398bef6eeb30522cc5` | #96 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/blueprint-runpod-learning-check` |
| `feature/blueprint-search-followup` | `6b29329d015b0c88374388361630f2035bfb4c93` | #105 merged (head match) | checked-out M1/M4 worktree; PR body #105 | **KEEP** | Keep remote ref |
| `feature/blueprint-seat-symmetry` | `fbb617082a8b0b66006f61b32ca9d57484dc6a45` | #109 merged (head match) | checked-out M1/M4 worktree; PR body #109 | **KEEP** | Keep remote ref |
| `feature/blueprint-showcase-tables` | `c24506ea751c0f4ce5e78c4c576b9db1d5642360` | #104 merged (head match) | checked-out M1/M4 worktree | **KEEP** | Keep remote ref |
| `feature/branching-training` | `f7e3c48c12307d140eb302bbbe29775bda396f6e` | #78 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/branching-training` |
| `feature/collector-branching` | `b666d8b45439593dc697447c78a9c3291e72fa28` | #77 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/collector-branching` |
| `feature/continuous-m4-baseline` | `3a31d35ab16aa31a33924448d0d3c03965710aee` | #88 merged (head match) | PR body #88 | **KEEP** | Keep remote ref |
| `feature/decision-errors` | `dc029c99b923f5065ae2e5cc0eb3b7edccdfe928` | No PR | unique/no PR review | **KEEP** | Keep remote ref |
| `feature/evaluation-arena` | `2831233f13c4f435aa119080abc0084cf0129bc4` | #44 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/evaluation-arena` |
| `feature/frozen-replay-fitting` | `88ccdf52332ba471f314ba6cd5c4d6211ac36cb4` | #72 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/frozen-replay-fitting` |
| `feature/heads-up-20bb` | `50afed7671cb72e7786e1d5fbe8788dad0d398eb` | #112 merged (head match) | checked-out M1/M4 worktree | **KEEP** | Keep remote ref |
| `feature/hu20-b100-diagnosis` | `3c2e6287fd106d8b61f3c6c7f406c9f2c238312c` | #117 merged (head match) | checked-out M1/M4 worktree | **KEEP** | Keep remote ref |
| `feature/hu20-b100-posterior-values` | `6b169903a861af3b055494948c24a7f512aa3b72` | #119 open (head match) | open PR head/base; checked-out M1/M4 worktree | **KEEP** | Keep remote ref |
| `feature/hu20-human-benchmark` | `82cac41e099c41d5a21e55037719d2ecc05cd61c` | #120 open (head match) | open PR head/base; PR body #120; release dependency #120 | **KEEP** | Keep remote ref |
| `feature/hu20-native-reopening` | `3dd54c77f1c0bb4107a638eccad1b03d80d6bb8f` | #115 merged (head match) | checked-out M1/M4 worktree | **KEEP** | Keep remote ref |
| `feature/hu20-reverse-lbr-acceleration` | `d073e55177bc7722e0aa72b35a87333ba719e467` | #121 open (head match) | open PR head/base; checked-out M1/M4 worktree | **KEEP** | Keep remote ref |
| `feature/hu20-training-scaling` | `7bee0d347c6d6f22fb9657fe7e6d64070bbae507` | #116 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/hu20-training-scaling` |
| `feature/ignore-paper-library` | `cf98727274b660134e9c256481739690c8289c42` | #99 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/ignore-paper-library` |
| `feature/local-cfr-conditional-diagnostic` | `b43fe2d282ed3eadb9df7b055fda6d4fe2582c20` | #107 merged (head match) | PR body #107 | **KEEP** | Keep remote ref |
| `feature/local-fullgame-training` | `e8714a7aa8dcae91e19f6813a32d8191a7e8418f` | #86 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/local-fullgame-training` |
| `feature/local-play-ui` | `db19f28715534a16bd6078273891dc7f405f2711` | #118 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/local-play-ui` |
| `feature/longer-holdem-results` | `03074d619203fd44626a913bcd562aa4674985c5` | #73 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/longer-holdem-results` |
| `feature/m4-next-training-analysis` | `d3477efc8d42b5753847be822a7a000e714ed625` | #87 merged (head match) | checked-out M1/M4 worktree | **KEEP** | Keep remote ref |
| `feature/m4-paper-training` | `414674e7b5a0e2e9c628af22c0044cd490e92a9b` | #89 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/m4-paper-training` |
| `feature/multistreet-campaign` | `c506c5eb0acdae665f04aa028a6f70715ba404e7` | #85 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/multistreet-campaign` |
| `feature/package-cleanup` | `e2fd919d963d7264ec3c01f4dc33739676c83ed1` | #64 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/package-cleanup` |
| `feature/persistent-critic` | `0e10f21298903420bb56bcb914e83610271cf4a2` | #76 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/persistent-critic` |
| `feature/player-observations` | `c0caaf22f463c3b624dff9c1ff6afbc8045f913d` | #42 merged (head match) | ROADMAP.md reference | **KEEP** | Keep remote ref |
| `feature/pluribus-blueprint` | `30cc306714a7648eb9376eeae2d776d7c396593f` | #91 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/pluribus-blueprint` |
| `feature/policy-comparison` | `7575de9dd1c3454b123fbf6233cf904766a8e2cd` | #74 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/policy-comparison` |
| `feature/postflop-replication` | `700cf0f159750ae2938ff03a93522fe9e8485f4c` | #110 merged (head match) | PR body #110 | **KEEP** | Keep remote ref |
| `feature/release-v0.4` | `fc5517e7e1aa276153dced46083b55848010589e` | Release-readiness PR pending | active M1 worktree and release candidate | **KEEP** | Keep remote ref |
| `feature/remove-legacy-workflows` | `c4e0bdd7cf69b0ad19077edd3b7a05f6012ad8d1` | #63 merged (head differs) | post-PR head drift | **KEEP** | Keep remote ref |
| `feature/river-cfr` | `cf549392c199331d4693f7f70401cbb7cdd7075c` | #108 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/river-cfr` |
| `feature/river-reference` | `d57c00a8b2f63b648b858611a49297f410b231ef` | #75 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/river-reference` |
| `feature/snapshot-readiness` | `642aeab163aba990619f21c8775b3a40b030fe6a` | #62 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/snapshot-readiness` |
| `feature/table-sessions` | `fd0818f04a52aacff05e4864b2a6e4136f0b2daa` | #43 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/table-sessions` |
| `feature/tabular-cfr` | `7889c3f3169c5ae84b6afafc4e8e159174d4ce7f` | #46 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/tabular-cfr` |
| `feature/three-player-20bb` | `bb20b834c655c0d4b21944a995c187986987dc93` | #113 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/feature/three-player-20bb` |
| `fix/day-paper-control-baseline` | `d9647e470ded5f66f2bb2ee584b431aa3f01539f` | #90 merged (head match) | No known live dependency | **DELETE after approval, backup and recheck** | `refs/backup/v04-20260930/fix/day-paper-control-baseline` |
| `main` | `19b7ae2c523e86e341bf2286df0f52795e1b21b7` | #36 merged (head differs), #11 closed (head differs) | default branch; open PR head/base; checked-out M1/M4 worktree; PR body #121,#120,#119,#118,#117,#116,#115,#114,#113,#112,#111,#110,#109,#107,#106,#104,#103,#102,#100,#98,#97,#95,#94,#93,#90,#89,#88,#87,#86,#85,#84,#83,#82,#81,#80,#79,#77,#76,#73,#70,#69,#68,#67,#66,#65,#63,#62,#61,#60,#59,#58,#57,#56,#55,#54,#53,#52,#50,#49,#48,#47,#46,#45,#44,#43,#42,#40,#39,#37,#32,#31,#30; post-PR head drift | **KEEP** | Keep remote ref |

Proposed exact branch DELETE allowlist (29 names):

```text
feature/20bb-robustness 848e958155b8fd91a6906f7f49fffa09041436f5
feature/benchmark-opponents bda25895e33d3255785df7b78110fcecd8f45b4c
feature/blueprint-averaged-extraction 349a6cb167ec5e1850b6300aa6a079f94803978e
feature/blueprint-history-coverage c509c1b400cd83f6e24a2257099d686e908fbc8a
feature/blueprint-mixed-table-demo 7d9fe4db594f0ad9ea7cb843342d7e5b82fe0fe2
feature/blueprint-postflop-search 7adde48ffeb5acac5278a0157628fd5902ab1a02
feature/blueprint-runpod-learning-check 5a3964b8e3ba7af0b5dc32398bef6eeb30522cc5
feature/branching-training f7e3c48c12307d140eb302bbbe29775bda396f6e
feature/collector-branching b666d8b45439593dc697447c78a9c3291e72fa28
feature/evaluation-arena 2831233f13c4f435aa119080abc0084cf0129bc4
feature/frozen-replay-fitting 88ccdf52332ba471f314ba6cd5c4d6211ac36cb4
feature/hu20-training-scaling 7bee0d347c6d6f22fb9657fe7e6d64070bbae507
feature/ignore-paper-library cf98727274b660134e9c256481739690c8289c42
feature/local-fullgame-training e8714a7aa8dcae91e19f6813a32d8191a7e8418f
feature/local-play-ui db19f28715534a16bd6078273891dc7f405f2711
feature/longer-holdem-results 03074d619203fd44626a913bcd562aa4674985c5
feature/m4-paper-training 414674e7b5a0e2e9c628af22c0044cd490e92a9b
feature/multistreet-campaign c506c5eb0acdae665f04aa028a6f70715ba404e7
feature/package-cleanup e2fd919d963d7264ec3c01f4dc33739676c83ed1
feature/persistent-critic 0e10f21298903420bb56bcb914e83610271cf4a2
feature/pluribus-blueprint 30cc306714a7648eb9376eeae2d776d7c396593f
feature/policy-comparison 7575de9dd1c3454b123fbf6233cf904766a8e2cd
feature/river-cfr cf549392c199331d4693f7f70401cbb7cdd7075c
feature/river-reference d57c00a8b2f63b648b858611a49297f410b231ef
feature/snapshot-readiness 642aeab163aba990619f21c8775b3a40b030fe6a
feature/table-sessions fd0818f04a52aacff05e4864b2a6e4136f0b2daa
feature/tabular-cfr 7889c3f3169c5ae84b6afafc4e8e159174d4ce7f
feature/three-player-20bb bb20b834c655c0d4b21944a995c187986987dc93
fix/day-paper-control-baseline d9647e470ded5f66f2bb2ee584b431aa3f01539f
```

`feature/local-play-ui` is eligible only because #118’s two raw screenshot URLs were changed to merged commit `19b7ae2c523e86e341bf2286df0f52795e1b21b7` and checked. Other branch-hosted PR-body references listed above remain protected. M4 also has detached worktrees; preserve their working files even when a remote branch is later deleted. Independent M4 clones for UI, scaling and #121 remain untouched.

## Tags and GitHub releases are separate objects

All six tags existed at this snapshot; **`v0.4.0` did not**. Only `0.3.3` is annotated (tag object `fb425432fd514a53e8a6c90d7db0aa6457890b33`, peeled commit `c2bdf6302d8d4481917b09500d20a3ade52ac14e`). The others are lightweight tags; the SHA below is both ref and commit. No version tag will be moved.

| Tag | Tag object/ref SHA | Peeled commit | Release ID/title/date | Assets and real metadata | Proposal |
| --- | --- | --- | --- | --- | --- |
| `release` | `11a4819bbbde480649f3d637ce00f2d26a4a809a` | `11a4819bbbde480649f3d637ce00f2d26a4a809a` | 203168000 / Trainable Beta 1.0 / 2025-03-01 | none | **KEEP; LABEL LEGACY** |
| `2.0` | `6324e2fb2d40f9635cada9fa0386bbcab9d68b44` | `6324e2fb2d40f9635cada9fa0386bbcab9d68b44` | 203585984 / release - 2.0 / 2025-03-04 | none | **KEEP; LABEL LEGACY** |
| `3.0` | `ab02ea6806cb72a0da153a1b875ed1130c5d46d8` | `ab02ea6806cb72a0da153a1b875ed1130c5d46d8` | 204613255 / 3.0 - Bet Calculation Fix & Opponent Modeling / 2025-03-09 | none | **KEEP; LABEL LEGACY** |
| `3.0.1` | `8bbc4efe43e5ba9bec46ac30dfbf6fe2fcce183b` | `8bbc4efe43e5ba9bec46ac30dfbf6fe2fcce183b` | 208959913 / 0.3.1 / 2025-03-28 | `deepcfr-poker-0.2.1.tar.gz` (53454 B; package metadata 0.2.1); `deepcfr_poker-0.2.1-py3-none-any.whl` (50022 B; package metadata 0.2.1) | **KEEP; CORRECT DISPLAY METADATA after owner approval** |
| `0.3.2` | `5dbe0f610f2ec8f5e6921124c8d448977ed67df6` | `5dbe0f610f2ec8f5e6921124c8d448977ed67df6` | 208976262 / 0.3.2 / 2025-03-28 | `deepcfr-poker-0.3.0.tar.gz` (53243 B; package metadata 0.3.0); `deepcfr_poker-0.3.0-py3-none-any.whl` (50043 B; package metadata 0.3.0) | **KEEP; CORRECT DISPLAY METADATA after owner approval** |
| `0.3.3` | `fb425432fd514a53e8a6c90d7db0aa6457890b33` | `c2bdf6302d8d4481917b09500d20a3ade52ac14e` | 346293558 / 0.3.3 / 2026-06-29 | none | **KEEP; LABEL LEGACY** |

`0.3.3` is currently Latest and has no assets. The older `release`, `2.0` and `3.0` tags mark published history, even where their titles overstate present capability. The `3.0.1` release is titled `0.3.1` but contains 0.2.1 wheel/sdist; `0.3.2` contains 0.3.0 wheel/sdist. Their bytes are genuine older packages. Proposed metadata correction means a visible legacy note explaining the mismatch, **not** retagging, renaming binaries or rewriting history. No tag, release or asset deletion is on the current allowlist. Published download counts, original release IDs and URLs should be preserved by default.

Every listed release is **published, regular (not draft or prerelease)**. Only release ID `346293558` (`0.3.3`) currently has the Latest flag. Each release URL is `https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/<tag>`; its body and GitHub API JSON are saved in the private backup snapshot. Asset IDs and independently downloaded byte hashes:

| Release | Asset ID | Original filename | SHA-256 |
| --- | ---: | --- | --- |
| `3.0.1` | 241593815 | `deepcfr-poker-0.2.1.tar.gz` | `22ca73983ba943611a79945ce255725ccfca8e72069881db4829f2c8956d84bd` |
| `3.0.1` | 241593813 | `deepcfr_poker-0.2.1-py3-none-any.whl` | `91ae85eaf20add335587b0bd4f911cbfbaebeef42899555a96daf8a9480d026c` |
| `0.3.2` | 241605625 | `deepcfr-poker-0.3.0.tar.gz` | `a2abd44a1cc85857b92623e9a19066f0fd54fb4caae4629426a08da3f47ddad5` |
| `0.3.2` | 241605618 | `deepcfr_poker-0.3.0-py3-none-any.whl` | `30f37b570eb7df27c6da424699bb92c68b47413dee5b06290fead164a66a225d` |

## Recovery and dry run before any approved deletion

1. Requery **all pages** of branches, PRs, releases and tags; compare each proposed branch SHA with this table. A changed ref, open dependency, new PR-body link, worktree or protection means stop and request fresh review. Check repository rulesets, tag protection and release immutability; do not disable them.
2. In durable private storage outside `/tmp`, create exact local backup refs `refs/backup/v04-20260930/<branch>` for each still-approved remote SHA. Bundle those refs and every tag object proposed for removal; run `git bundle verify`, clone/fetch a test copy, and record SHA-256 plus an inventory manifest. A SHA alone cannot rescue an unreachable object. For releases/assets proposed for removal, save the API JSON and original uploaded bytes with SHA-256 first; Git bundles do not contain uploads. A preliminary **115,402,284-byte** bundle of the 29 proposed branch heads was created at `/Users/dberweger/Local/release-backups/v0.4.0-20260930/remote-branch-candidates.bundle`, SHA-256 `f1650e5fec44a2e377e82b717250fbfc24078ef97048184e488643b557089b63`, verified with `git bundle verify`, and cloned into a test bare repository with all 29 backup refs present. The private `backup-manifest.json` records exact SHA-256 values for the bundle, release API snapshots and four downloaded legacy assets. Refresh it if any remote SHA changes before approval.
3. Dry run each exact ref: `git ls-remote origin refs/heads/NAME`, compare full SHA, verify backup bundle and approved allowlist, then prepare `git push origin --delete NAME` **one name at a time**. Do not execute until separate owner approval. Never use wildcards, pruning or force push. No local worktree or branch deletion is part of this plan.
4. Recreated branches and Git objects can be restored from the bundle if kept. Recreated releases/assets may have new IDs, URLs, dates and download counts. Legacy release metadata edits are reversible text changes, but links may cache. Post-change verification must check refs, downloads and all linked screenshots.

## Local and file cleanup

M1 primary, release, research, scratch and Claude worktrees; M4 primary plus five registered worktrees; independent M4 UI/scaling/#121 clones are in use or evidence-bearing. **KEEP all.** No local worktree pruning or checkpoint/journal/result deletion is proposed.

| File or group | Decision | Reference check and reason |
| --- | --- | --- |
| `readme.md` | DEPRECATE old front-page prose in place | Linked by ROADMAP, package metadata and GitHub; preserve lowercase filename on case-insensitive macOS. Replace stale quick start, retain research through index. |
| `docs/play-web.md` | KEEP and update | Current web service and browser guide; replace private M4 quick-start paths while preserving historical smoke provenance. |
| `scripts/play_hu20_native.py`, `scripts/play_hu20.py`, `src/play_api/`, `src/game/`, `src/blueprint/` | KEEP | Active CLI/API, inference, engine, observation, settlement and replay paths; tests and docs reference them. |
| `tests/`, `configs/`, `.github/workflows/tests.yml`, `requirements*.txt` | KEEP | CI, CLI entrypoints and reproducibility depend on them; includes historical regression tests. |
| `docs/reports/`, manifests, JSONL, model cards, old neural code | KEEP | Sealed evidence, historical negative results, hashes and still referenced research. No content move or manifest invalidation. |
| `.gitignore`-excluded `results/`, `models/`, checkpoints, journals and coordination files | KEEP locally; never bundle | Private and research state; absence from ordinary Git is intentional. |
| Tracked generated outputs / duplicate launchers | No REMOVE candidate after audit | `git ls-files` found no tracked cache, `.pyc`, service journal, checkpoint or release wheel; current launchers are referenced by CI/docs/tests. Lack of direct imports alone was not treated as dead code. |

No file deletion is proposed in the release-readiness PR. This is a conservative cleanup of the public entry point and remote-ref proposal, preserving evidence.
