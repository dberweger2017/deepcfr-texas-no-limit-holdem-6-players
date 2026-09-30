# DeepCFR Poker AI

**v0.4 — Rebuilt Poker AI Research Preview**

Play, inspect and reproduce a learning poker agent. This project began with neural Deep CFR; its featured playable policy now uses **tabular external-sampling CFR**. The local table lets a person play the fixed-first-seed B100M policy, trained to 100 million *traversal nodes*. Strong six-player poker remains the destination, not a capability of this release.

![The local heads-up poker table](docs/play-web/restricted-table.png)

## What runs today

| Path | Game and status |
| --- | --- |
| [Local web table](docs/play-web.md) and [terminal play](scripts/play_hu20_native.py) | **Featured:** two-player no-limit Hold'em, 20 BB (2,000 chips) per seat reset each hand, no rake or ante. The B100M inference export is a saved experimental policy. |
| [Three-player research](docs/tp20-model-card.md) | Separate TP20 artifacts and experimental results; not supported by the web table or B100M. |
| [Scripted multiplayer sandbox](docs/benchmarks.md) | Exercises four-to-six-player rules, sessions and evaluation. It is not a trained six-player agent. |
| [Six-player, 100 BB research](docs/research-history.md) | Historical neural and blueprint experiments; the v0.5 strength criterion remains unmet. |

The v0.4 source workflow uses Python 3.11 with a pinned Rust-backed [pokers engine](https://github.com/dberweger2017/pokers/tree/5db20e3d5d6862b32a7402035c1340b622d3b005). The web table runs **locally on loopback**, with an access token. It is not a hosted poker service. We have not validated every operating system or published a new PyPI package.

## Quick start

Install Python 3.11, Rust/Cargo, Git and a C/C++ build toolchain. The native engine is pinned in [requirements.txt](requirements.txt) to audited commit `5db20e3d5d6862b32a7402035c1340b622d3b005`. From a clean checkout of the **published `v0.4.0` tag**:

```sh
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git
cd deepcfr-texas-no-limit-holdem-6-players
git checkout v0.4.0
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir models
curl -fL --output models/B100M-HU20-current-seed-2026093001.json.gz \
  https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/download/v0.4.0/B100M-HU20-current-seed-2026093001.json.gz
python -m scripts.verify_v04_model models/B100M-HU20-current-seed-2026093001.json.gz
python -m src.play_api.server \
  --policy models/B100M-HU20-current-seed-2026093001.json.gz \
  --data-dir results/play-web --source-version "$(git rev-parse HEAD)"
```

The verifier checks **40,144,034 bytes** and SHA-256 `4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf` before model loading. The service independently checks the hash, HU20 game/schema, two seats, uncapped restricted menu and current-policy extraction. Do not substitute a training checkpoint or an earlier B20M export. The download becomes available when the v0.4.0 release is published; during review use the verified local artifact described in the [model card](docs/releases/v0.4.0/MODEL_CARD.md).

`mkdir models` intentionally fails when that directory already exists, so the example cannot silently overwrite a previously downloaded model. Choose a fresh destination or inspect existing files before rerunning it.

Open `http://127.0.0.1:8765/`. Read `results/play-web/access.token` locally and enter it in the table. Keep the token out of URLs, screenshots and shared logs. Choose a play mode, create a session and deal a hand. The server stays on `127.0.0.1`, including when reached through an [SSH tunnel](docs/play-web.md#model-and-launch). Refresh resumes the same session; lost responses can be retried without a second wager. [Full play and replay guide](docs/play-web.md).

## Play modes and human sessions

**Restricted research** shows the concrete actions in the model's trained menu. **Free sizing · experimental** accepts any native-legal integer-chip raise-to amount. The native engine executes that exact wager, while the bot still uses its restricted menu and existing missing-key fallback. This interface does not solve off-tree strategy. Both modes keep hidden cards and private randomness on the server.

The base table journals completed hands and supports native replay. The separate [human benchmark framework in PR #120](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/120) is **not included until reviewed and merged into this release source**. If included, it records raw human session results, exact planned hand counts, abort status and sanitized exports. It neither establishes human playing strength nor implements AIVAT.

## Measured research results

The featured first-seed B100M is **one saved inference export**, not the three-lineage aggregate below. In the frozen [#116 HU20 scaling study](docs/reports/hu20-scaling-m4-recovery.md), three uncapped 20 BB lineages continued from 20M to 100M traversal nodes. On paired two-position schedules, the **aggregate** 100M-minus-own-20M target profit improved **+25.94 BB/100 [two-sided 97.5% interval +7.43, +44.45]** against original-cap2 bounded local best response (2,048 paired blocks per seed/checkpoint) and **+19.39 [+2.86, +35.93]** against native-legal pressure (4,096 blocks). Absolute 100M profit against that bounded attacker remained **−73.14 BB/100**. The attacker is limited, not an exact exploitability certificate.

The separate [completed #116 diagnostics](docs/reports/hu20-scaling-diagnostics.md) found improvements against minraise and passive controls relative to each B lineage's own 20M policy, but small secondary panels were mixed. Pot-pressure profit was only +3.29 BB/100 in a 256-block-per-policy panel, with frequent fallback after off-menu histories. [#117's diagnosis](docs/reports/hu20-b100-diagnosis-m4.md) used a **different** fresh 512-block schedule and must not be pooled with #116 estimates. Independent late-street trained coverage stayed thin (first-seed river 56.2% on a 4,248-observation fixture). These are research outcomes, not a claim that the released seed beats human players.

## Limits and next milestones

B100M supports only its specified heads-up 20 BB, no-rake/no-ante game. Each hand resets stacks; this is not tournament play. Free sizing can push histories outside the bot's trained abstraction. No six-player or 100 BB strength follows from these results, and the older neural Deep CFR work is not a complete Pluribus reproduction. Historical failures and the full trail remain in the [research index](docs/research-history.md) and [roadmap](ROADMAP.md).

### What 1.0 means

The [formal v0.5 and v1.0 criteria](ROADMAP.md#release-milestones) remain unchanged. v0.5 requires reliable positive six-player 100 BB profit against the scripted pool on fresh held-out deals across at least two independent seeds. v1.0 requires a credible professional reference benchmark, predeclared confirmation, multiple training seeds, legal information-safe play and complete reproducibility. v0.4 qualifies neither milestone.

## Developers and licenses

Run the suite with `python -m pip install -r requirements-dev.txt` followed by `python -m pytest -q`. [CI](.github/workflows/tests.yml) also checks observation, sessions, arena, solver and neural baseline. Read the [rules](docs/rules.md), [observation contract](docs/observations.md), [web guide](docs/play-web.md), and [release notes](docs/releases/v0.4.0/RELEASE_NOTES.md).

This repository's own code is [MIT](LICENSE.txt). The pinned `pokers` fork and its upstream do **not currently publish a license grant** in their repositories or package metadata. Engine redistribution terms need clarification before public release; the v0.4 bundle does not include an engine binary. Historical reports and referenced papers retain their own rights.
