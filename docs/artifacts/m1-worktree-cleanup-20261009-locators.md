# M1 cleanup source and archive locators — October 9, 2026

57 retired worktrees; measured removal-batch gain 17,428,353,024 bytes; 30,322,909,184 bytes free immediately after cleanup. [Guide](m1-worktree-cleanup-20261009.md) · [complete original-path/member/stat receipt](https://drive.google.com/file/d/1J1OsjpghvHUmhxSEnmdkpnzqChKSCNmb/view?usp=drivesdk). The complete receipt is 7,561,003 bytes, SHA256 `edf33765bf1ac80faec1da8419bfbca79dca5c642078aa935c431b3e94d41379`.

## Source restore map

Branches remain. Every source revision below is pinned at `refs/cleanup/m1-worktrees-20261009/<source SHA>` in the M1 shared Git repository. To restore a checkout without changing another branch, substitute one row's exact path and source SHA in:

```sh
git -C /Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players worktree add --detach '<original absolute path>' 'refs/cleanup/m1-worktrees-20261009/<source SHA>'
```

| Original absolute path | Preserved branch | Source SHA | Owning evidence PR |
|---|---|---|---|
| `/private/tmp/claude-501/-Users-dberweger-Local-deepcfr-texas-no-limit-holdem-6-players--claude-worktrees-deepcfr-repo-access-830ea6/0d280ecb-f629-4f5f-8e26-12f6ed1f4330/scratchpad/arena-wt` | `feature/hu20-turn-search-arena` | `f6c58cc8c9d7581f9edcfa11d77ccd7292cf407e` | Source/build only |
| `/private/tmp/m4-drive-organization-20261005/repo` | `feature/m4-drive-artifact-index` | `d2dc9a7c1788eca3508f5b4c811dcc9d25f2ad88` | Source/build only |
| `/Users/dberweger/.codex/worktrees/7a0f/deepcfr-texas-no-limit-holdem-6-players` | `feature/hu20-board-pooling` | `284c24544c70186e7a6e3ef1a982059590041879` | Source/build only |
| `/Users/dberweger/.codex/worktrees/c264/deepcfr-texas-no-limit-holdem-6-players` | `detached` | `60f516da6b24563a49e609636a31fa17ae719bde` | Source/build only |
| `/Users/dberweger/.codex/worktrees/hu20-500m-campaign/deepcfr-texas-no-limit-holdem-6-players` | `feature/hu20-500m-campaign` | `396747125e810da1ef9be4fa11d85ef5bfef0e9e` | Source/build only |
| `/Users/dberweger/.codex/worktrees/hu20-exact-lbr-ranker/deepcfr-texas-no-limit-holdem-6-players` | `feature/hu20-exact-lbr-ranker` | `b49c1d9b231bd3f298e89a89f7d848a26fef4de1` | Source/build only |
| `/Users/dberweger/.codex/worktrees/hu20-history-2x2/deepcfr-texas-no-limit-holdem-6-players` | `feature/hu20-history-2x2` | `6cffc7290a5d2a5f3fb4e9567d872662a130ab5c` | Source/build only |
| `/Users/dberweger/.codex/worktrees/local-cfr-conditional-diagnostic` | `feature/hu20-native-reopening` | `3dd54c77f1c0bb4107a638eccad1b03d80d6bb8f` | Source/build only |
| `/Users/dberweger/.codex/worktrees/posterior-audit-v2/deepcfr-texas-no-limit-holdem-6-players` | `feature/hu20-posterior-audit-v2` | `78d4a793bfe61a685b38e2cab85c68a4537fc540` | Source/build only |
| `/Users/dberweger/.codex/worktrees/posterior-values/deepcfr-texas-no-limit-holdem-6-players` | `feature/hu20-b100-posterior-values` | `6b169903a861af3b055494948c24a7f512aa3b72` | Source/build only |
| `/Users/dberweger/.codex/worktrees/results-drive-archive/deepcfr-texas-no-limit-holdem-6-players` | `feature/research-final-inventory-20261004` | `30dba1efc78c7879405ada0b090c1b2b2236892e` | Source/build only |
| `/Users/dberweger/.codex/worktrees/reverse-lbr-acceleration/deepcfr-texas-no-limit-holdem-6-players` | `feature/hu20-lbr-kernel-profile-report` | `970cb064106ad6b0ef4499855b5c16eda77d25ba` | Source/build only |
| `/Users/dberweger/.codex/worktrees/river-cfr/deepcfr-texas-no-limit-holdem-6-players` | `feature/m4-next-training-analysis` | `d3477efc8d42b5753847be822a7a000e714ed625` | Source/build only |
| `/Users/dberweger/.codex/worktrees/runpod-hu20-parity/deepcfr-texas-no-limit-holdem-6-players` | `feature/runpod-hu20-parity` | `ce87da3450f1310587bd78b0243967b497eb1867` | Source/build only |
| `/Users/dberweger/.codex/worktrees/runpod-mature-cpu/deepcfr-texas-no-limit-holdem-6-players` | `feature/runpod-mature-cpu` | `256b38224bece066d9e5073b5b05464e30fd8e02` | Source/build only |
| `/Users/dberweger/.codex/worktrees/trainer-observation-reuse/deepcfr-texas-no-limit-holdem-6-players` | `feature/trainer-observation-reuse` | `e7aa2ea32cdfc9b1418b8f0d6714500e22e0e7b3` | Source/build only |
| `/Users/dberweger/.t3/worktrees/deepcfr-texas-no-limit-holdem-6-players/feature-native-hu100-preparation` | `feature/native-hu100-preparation` | `cc18d20e413a8ca08eeb9e755786a41b0c2db17e` | Source/build only |
| `/Users/dberweger/.t3/worktrees/deepcfr-texas-no-limit-holdem-6-players/t3code-32636be5` | `t3code/review-native-hu20-bench` | `13e9db91fe1a48567674a8ac752bf3831bcfa9a6` | Source/build only |
| `/Users/dberweger/.t3/worktrees/deepcfr-texas-no-limit-holdem-6-players/t3code-ba4e7725` | `feature/o-learning-curve` | `62ff6a564c99810830abe012a7cdf5b80d3992d4` | Source/build only |
| `/Users/dberweger/Local/blueprint-showcase-tables` | `feature/blueprint-showcase-tables` | `c24506ea751c0f4ce5e78c4c576b9db1d5642360` | Source/build only |
| `/Users/dberweger/Local/deepcfr-hu20-floor-control` | `feature/hu20-floor-control` | `3917fb647fd12af157721b53b114ba4ef1fa7e47` | Source/build only |
| `/Users/dberweger/Local/deepcfr-main-scratch` | `worktree/main-scratch` | `e9af36c04ac1e4323a3ec6581385d291aba6b941` | Source/build only |
| `/Users/dberweger/Local/deepcfr-necessary-cleaning` | `feature/necessary-cleaning` | `088df365ffb838b083b39468f1d63671bf110ca2` | Source/build only |
| `/Users/dberweger/Local/deepcfr-o-10b` | `feature/hu20-o-10b` | `cddd66219ee771048e099fd2934deb73c36e65d7` | Source/build only |
| `/Users/dberweger/Local/deepcfr-review-pr178` | `feature/review-zero-mass-current` | `1ec697e855cba95019362f9980017071317aa12e` | Source/build only |
| `/Users/dberweger/Local/deepcfr-robustness-measure` | `detached` | `36a678ae4519d578f5d89ed031b37b3c517a8b3f` | Source/build only |
| `/Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players/.claude/worktrees/agent-a95c88c1f98badca4` | `worktree-agent-a95c88c1f98badca4` | `13e9db91fe1a48567674a8ac752bf3831bcfa9a6` | Source/build only |
| `/Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players/.claude/worktrees/curious-jumping-donut` | `worktree-curious-jumping-donut` | `e9af36c04ac1e4323a3ec6581385d291aba6b941` | Source/build only |
| `/Users/dberweger/Local/deepcfr-v041-o-confirmation` | `feature/hu20-v041-o-confirmation` | `f19f1dc59965974043b49dd03c55374ecb44dd91` | Source/build only |
| `/Users/dberweger/Local/deepcfr-v041-release` | `feature/v041-release` | `06944412ac393480f378c0ca21172aa6a2855bcc` | Source/build only |
| `/Users/dberweger/Local/hu20-equity-buckets-closeout-repo` | `feature/hu20-equity-buckets-closeout` | `6339eb6f4c817c647829d354625b948fe55d8db3` | Source/build only |
| `/Users/dberweger/Local/hu20-trainer-bench-closeout-repo` | `feature/hu20-trainer-bench` | `44c6ff5aedb5730dce7de592119526edf303dd8d` | Source/build only |
| `/Users/dberweger/Local/m1-storage-vacuum-20261007/repo` | `feature/m1-storage-vacuum-20261007` | `4d55771f6c47e799d14db38edc4e3f6391888f6b` | Source/build only |
| `/Users/dberweger/Local/m4-storage-vacuum-20261007/repo` | `feature/m4-storage-vacuum-20261007` | `66bb819a50f5e2578af71aea9ad2102be318fd7f` | Source/build only |
| `/Users/dberweger/Local/m4-uploaded-cleanup-evening-20261006` | `feature/clean-uploaded-m4-research` | `fd5b4ff804b06e80c52e4eba5e76cb21aa3f1cb7` | Source/build only |
| `/Users/dberweger/Local/merged-research-cleanup-20261006` | `feature/clean-merged-research-originals` | `9d7d0db0b42dcfa812ca567ebbf44b6d0c20e764` | Source/build only |
| `/Users/dberweger/Local/pr169-rereview-b60b3ff-20261005` | `detached` | `b60b3ff9c5c4a8820575d3afae5c78df2f5e6d7b` | Source/build only |
| `/Users/dberweger/Local/pr169-review-13616df-20261005` | `detached` | `13616dfe9fd8fc2b3a809f43c76b582a6fdc1290` | Source/build only |
| `/Users/dberweger/Local/pr169-review-20261005` | `detached` | `1358004f026b9f711cbfdc8d5a8fece04c664444` | Source/build only |
| `/Users/dberweger/Local/pr169-review-b12dc4a-20261005` | `detached` | `b12dc4a40518f2d4d42b5106360eb0d3b22a9b2d` | Source/build only |
| `/Users/dberweger/Local/pr171-cfr-plus-arena` | `detached` | `6c8b49a059c80bd942150f26bc25847aa830917d` | Source/build only |
| `/Users/dberweger/Local/pr171-cfr-plus-confirmation` | `feature/hu20-cfr-plus-confirmation` | `d69aa1fa2d8a610eca8d65a7a37524ad4be614f7` | Source/build only |
| `/Users/dberweger/Local/pr171-cfr-plus-direct-nine` | `detached` | `c471678b563982e926ec9256a0a00fedb5d0f497` | Source/build only |
| `/Users/dberweger/Local/pr171-cfr-plus-review` | `detached` | `60f516da6b24563a49e609636a31fa17ae719bde` | Source/build only |
| `/Users/dberweger/Local/storage-ignore-fix-20261006` | `feature/ignore-research-model-artifacts` | `d0a8a40e771ebc0f463f052d4261d1438b7002c2` | Source/build only |
| `/Users/dberweger/.codex/worktrees/blueprint-local-cfr/deepcfr-texas-no-limit-holdem-6-players` | `feature/blueprint-seat-symmetry` | `fbb617082a8b0b66006f61b32ca9d57484dc6a45` | #109 |
| `/Users/dberweger/.codex/worktrees/postflop-replication/deepcfr-texas-no-limit-holdem-6-players` | `feature/heads-up-20bb` | `50afed7671cb72e7786e1d5fbe8788dad0d398eb` | #112 |
| `/Users/dberweger/.t3/worktrees/deepcfr-texas-no-limit-holdem-6-players/feature-hu100-checkpoint-learning-curves` | `feature/hu100-checkpoint-learning-curves` | `acef17133bad6f26c4b2265772e64f40abe9a9fd` | #200 |
| `/Users/dberweger/Local/blueprint-search-followup` | `feature/blueprint-search-followup` | `6b29329d015b0c88374388361630f2035bfb4c93` | #105 |
| `/Users/dberweger/Local/deepcfr-global-bucket-validation` | `feature/hu20-global-bucket-validation` | `c031b6212852bf520e38faf5c5d54b2d65763eda` | #190 |
| `/Users/dberweger/Local/deepcfr-v042-release` | `feature/v042-publication-record` | `729695ea3b4b7bb46f6fcc058d85b50cb445d981` | #199 |
| `/Users/dberweger/Local/hu100-coverage-loss-diagnosis-20261008` | `feature/hu100-coverage-loss-diagnosis` | `19101580ba04b45c96d5602ac10715279e293d2e` | #201 |
| `/Users/dberweger/Local/hu100-export-audit-memory-20261008` | `feature/hu100-export-audit-memory` | `021e3c9077302b9c49a49481f25f3c8a66e5f66d` | #202 |
| `/Users/dberweger/Local/luna-shield-chrome` | `feature/luna-shield-chrome` | `f835d203b83fa808f1dd9eb97a47532d229d25ca` | #174 |
| `/Users/dberweger/Local/pr166-hu20-closeout` | `feature/hu20-turn-search-closeout` | `a253c1b00e9a505263815da783e2469dab023196` | #166 |
| `/Users/dberweger/Local/v042-lbr-confirmation` | `feature/v042-lbr-confirmation` | `2731a39070c45042736215820622944a0b09e745` | #188 |
| `/Users/dberweger/Local/deepcfr-web-spectator` | `feature/web-spectator` | `23ef617a753fdbbf4d4cd770822c7e10a026239e` | #193 |

## Accepted archive map

All snapshots embed `ARCHIVE-MANIFEST.json`; its SHA256 is in the table. Native uploaded=1, uploading=0, conflicts=0, trashed=0 and exact document size passed for every archive, with separate cloud ID/name/size/parent and owner-only permission confirmation. No remote archive bytes were downloaded. All six reused PR190/PR202 archives have current native/cloud acceptance in the complete receipt; old payload hashes were not audited again.

| PR | Archive | Bytes | Archive SHA256 | Manifest SHA256 | New /reused original members |
|---|---|---:|---|---|---:|
| #109 | [m1-retirement-pr109-20261009.zip](https://drive.google.com/file/d/1hFCVwlPxq0NZea7jMvjRY_nkEWt81cYB/view?usp=drivesdk) | 1,500 | `4d73596dfca0c5b6d2e24e08b5446d1db1eb39591be95faffb73bd6d81c5acac` | `22078d4fb3d768c0881e25860904863cd797831473d4e4a55d1f9c6a0f6dabdf` | 1 /0 |
| #112 | [m1-retirement-pr112-20261009.zip](https://drive.google.com/file/d/1Tag_3_GXhEf_xMZ4RpxPe_26UPd8HjTr/view?usp=drivesdk) | 17,474,323 | `c0b052ac809e6a1872939b655248335b84e4a02e3b62856a896362a03815ecf4` | `44c9859a286ef11f3192c3a8a16baf01b6844793fb4eea2db2fc097905eee810` | 3 /0 |
| #200 | [m1-retirement-pr200-20261009.zip](https://drive.google.com/file/d/1wVRs9F7EmjxVyOyQ414CsU-v7tNAxPd2/view?usp=drivesdk) | 649,870 | `20ef52888581f76e1d88c5bbd4a53913284c76b364965e313dda6ca531614745` | `bcbf4a624ac4fd76d79ad65896bc8d932f34c651b08b35e7d2fbc34028522edc` | 21 /0 |
| #105 | [m1-retirement-pr105-20261009.zip](https://drive.google.com/file/d/1jwgbMXtgPj4259mEsEygDBqSGce7IfVL/view?usp=drivesdk) | 1,544 | `6f9cf5375ac6a6e873f074b1d4e59d751edc15cf35ffb444f861a859eaf7b14b` | `6b233027844f4a2f9f514480b5ac82ff5edc011bd2d608785262a203528efa53` | 1 /0 |
| #190 | [m1-retirement-pr190-20261009.zip](https://drive.google.com/file/d/1yPKWhlg18LNS03Ykm3oxs9NlKUyct2U2/view?usp=drivesdk) | 843,760 | `8262edac135914c9739ea7eaf10c67795a53739f00b9a695fadcea9803884bf9` | `b752cc8fc74eb8eee2a5816b836e1861a1a84c50f0276ac89512da9e824f0ce2` | 15 /2934 |
| #199 | [m1-retirement-pr199-20261009.zip](https://drive.google.com/file/d/1mpBNQLGeOWqeUjBRRdPWzjDB2sHiW51A/view?usp=drivesdk) | 253,345,864 | `9e501b9882c0a00777e6fcabf53c7611105553f0a6093e6dc5698c03862826f4` | `9db15b3ad6ca78fbbca035ba7153cd0b47a37722a289be9f6ca5c1d720faab98` | 46 /0 |
| #193 | [m1-retirement-pr193-20261009.zip](https://drive.google.com/file/d/1a31dazuAS1mzvbj9sG4JXO83gs2cBpBW/view?usp=drivesdk) | 188,162,257 | `139c73833112a6c936d46748fab060b1bc4e40c721f6a765d3bd0ed88d43c746` | `c8fb5d4cf632e3dc56f66a67d95a1310365e80bd18ee95e7c543a7983701b824` | 37 /0 |
| #201 | [m1-retirement-pr201-20261009.zip](https://drive.google.com/file/d/1rB70ruysjH9jpz5ixO5gO1W_v_0eqTZd/view?usp=drivesdk) | 11,526,238 | `38996ba5e081cbbe20f691a6178283e172d9fcc174075612417b9689a855fccb` | `3e4777c1d2c1fdec1e21c23d06bf27ebfb90fecb6d86b89e944d16df3716c619` | 347 /0 |
| #202 | [m1-retirement-pr202-20261009.zip](https://drive.google.com/file/d/1HO96dGJ9Zcf49ftsR8pAVKgYIpEVVzB2/view?usp=drivesdk) | 158,035 | `03c41fc3d9175f04cf72d72f4fd9cc387da1d5b3bae4c01931f3c1a5fabd2c76` | `dcb463911c34554fc3dca2d8cf5dc766f82fb34bf7bb571d366dbb17cbdce9fe` | 11 /12 |
| #174 | [m1-retirement-pr174-20261009.zip](https://drive.google.com/file/d/1VMmc8AsL1Aku17oJtTcGkmjRExmjcNCM/view?usp=drivesdk) | 144,836,983 | `b27bd27eecddec1ef5296a77ab2d2fac83224bfdeffe19947bad4736ffcad9cc` | `5283662c23185797f8f67e33d6f5787faa0619e3fb32ccfab555d52f9f520a95` | 24 /0 |
| #166 | [m1-retirement-pr166-20261009.zip](https://drive.google.com/file/d/1OMPMLY4o8pNjjcZUgocsUuq7pLttMxQf/view?usp=drivesdk) | 429,372 | `7d208f0c8a758bef402e480af6d91bf121dfbb7abdcf6e7db1ad7b7d9ca98f4e` | `5ea12882d6a6ba461d877eeddf34aea9c18d49ae66128cf58807e2abf6e36b34` | 11 /0 |
| #188 | [m1-retirement-pr188-20261009.zip](https://drive.google.com/file/d/1LpUahSjRwWNDFcZMdDSnQvWoMJxi5Nis/view?usp=drivesdk) | 2,838,542 | `c34d29e3421a614eaa581b435a83e85e83e6ccc2ed1868214468aad5233028b1` | `b12c3dac07cd75ae4c5dd9890726495b1b0f13dd4b434073ada465bbedc46888` | 114 /0 |

| PR | Designated native folder under `~/Local/Research-Cloud/` | Drive folder |
|---|---|---|
| #109 | `PR-109-seat-symmetry` | [1PfI-YKmE0LXTfFt4vMHovkveLF7s3woP](https://drive.google.com/drive/folders/1PfI-YKmE0LXTfFt4vMHovkveLF7s3woP) |
| #112 | `PR-112-HU20` | [16XG3GqF4vflkcsqvika0ZYideXUjSw_W](https://drive.google.com/drive/folders/16XG3GqF4vflkcsqvika0ZYideXUjSw_W) |
| #200 | `PR-200-HU100-learning-curves` | [12azuxRTXRyEnpO-7tSb6TTOA4teHccMf](https://drive.google.com/drive/folders/12azuxRTXRyEnpO-7tSb6TTOA4teHccMf) |
| #105 | `M4-blueprint-search-followup` | [1rhGV45mqLEG0zosI4LnhetD8zHKv2fB2](https://drive.google.com/drive/folders/1rhGV45mqLEG0zosI4LnhetD8zHKv2fB2) |
| #190 | `PR-190-HU20-bucket-validation` | [1C0z2LwrqiUZgVwU_flkBwvxM0MV5LeDh](https://drive.google.com/drive/folders/1C0z2LwrqiUZgVwU_flkBwvxM0MV5LeDh) |
| #199 | `PR-199-v0.4.2-publication` | [18igPOSguwguT9fRQtSzBi5HCKacIIKkQ](https://drive.google.com/drive/folders/18igPOSguwguT9fRQtSzBi5HCKacIIKkQ) |
| #193 | `PR-193-web-spectator` | [1PnIKJj4penks16uQrxWuDqaChJqNqw36](https://drive.google.com/drive/folders/1PnIKJj4penks16uQrxWuDqaChJqNqw36) |
| #201 | `PR-201-HU100-coverage-loss-diagnosis` | [1EhGCzqqJf_pmzv5comk2jMmDpHks0QFI](https://drive.google.com/drive/folders/1EhGCzqqJf_pmzv5comk2jMmDpHks0QFI) |
| #202 | `PR-202-HU100-export-audit-memory` | [10d6DPp8i6uQrW5QaEaJEW364Dd7Vt7nv](https://drive.google.com/drive/folders/10d6DPp8i6uQrW5QaEaJEW364Dd7Vt7nv) |
| #174 | `PR-174-Luna-Shield/20261006` | [14_bys-Ib_lHUtnuoZDyV5AfkVuOF-pVo](https://drive.google.com/drive/folders/14_bys-Ib_lHUtnuoZDyV5AfkVuOF-pVo) |
| #166 | `PR-166-HU20-turn-search-arena` | [1iVCttvjpYo8X4jD9C3tcflLyT_Y9QBqO](https://drive.google.com/drive/folders/1iVCttvjpYo8X4jD9C3tcflLyT_Y9QBqO) |
| #188 | `PR-188-HU20-O-10B-LBR` | [1mv9v1VV2-Szfn8jNAjvpU4oWvRdqIkY2](https://drive.google.com/drive/folders/1mv9v1VV2-Szfn8jNAjvpU4oWvRdqIkY2) |

## Retained dependencies

The two otherwise inactive evidence roots still referenced by maintained configuration/runtime source were retained:

- `/Users/dberweger/.codex/worktrees/blueprint-averaged-extraction/deepcfr-texas-no-limit-holdem-6-players` — Referenced by maintained runtime source or configuration.
- `/Users/dberweger/Local/native-recovery-hu100-20261008` — Referenced by maintained runtime source or configuration.

Other recent, active, dirty or uncertain worktrees were outside the eligible set. Active/open PR inputs, shared Git and synced files remain intact. The detailed receipt records every original stat, restore member, excluded disposable file and temporary nonsynced archive copy; the guide explains restoration and the measurement limits.
