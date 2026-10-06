# A local heads-up poker research bot

v0.4.1 plays a tabular, linearly weighted opponent-sampled CFR average trained for 1B nodes. The game is two-player no-limit Hold'em at 20 BB, with no rake or ante and fresh stacks each hand. The local web table supports replayable sessions, private bot cards, a restricted research menu and experimental free human bet sizing.

Against shipped v0.4.0, the three retained v0.4.1 lineages win **+10.50 [7.90, 13.10] BB/100**; the selected first seed wins **+11.12 [7.85, 14.39]**. An independent earlier match measured **+11.45 [8.83, 14.07]**. A bounded LBR attacker earns **28.17 [17.93, 38.42]** against O versus **65.73 [56.84, 74.62] BB/100** against the v0.4.0 lineages, so lower is better. On fixed turn/river spots, first-seed best-response gain is **1.0665 [1.0023, 1.1315]** versus **2.7147 [2.4922, 2.9323] BB**. All intervals are 95%; scopes and the full thirteen-panel table are in the [v0.4.0 → v0.4.1 report](docs/reports/v0.4.1-release.md). These are measured improvements, not a full-game Nash certificate or a human-strength claim.

## Install and play

Use Python 3.11, Git and the GitHub CLI. From a fresh checkout:

```sh
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git
cd deepcfr-texas-no-limit-holdem-6-players
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir models
gh release download v0.4.1 --pattern 'O1B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz' --dir models
gh release download v0.4.0 --pattern 'B100M-HU20-current-seed-2026093001.json.gz' --dir models
python -m scripts.verify_v041_model models/O1B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz
python -m scripts.verify_v04_model models/B100M-HU20-current-seed-2026093001.json.gz
python -m src.play_api.server --models-dir models
```

The v0.4.1 download command works after the owner-approved release is published; the release PR uses verified staged assets until then. `mkdir models` and downloads fail rather than overwrite existing artifacts. If the directory exists, inspect it or use a fresh destination. Installation builds the pinned native `pokers` engine; follow its build error instructions if your platform needs a Rust toolchain.

Open **http://127.0.0.1:8765/** and enter the token from `results/play-web/access.token` locally. Keep tokens out of URLs and screenshots. Choose **v0.4.1** (default) or **v0.4.0**, choose bet sizing, create a session and deal. Seats alternate each hand. Model choice is frozen for that session; refresh resumes it, including after server restart. Use a new session to change models. Both pinned files are required for model selection; a missing or corrupt default causes an error, never a silent replacement. The two loaded tabular readers need several GiB of RAM.

Restricted mode uses the concrete training menu. Free sizing executes any native-legal integer-chip human wager exactly; the bot keeps its trained menu and unchanged uniform fallback on missing/zero-mass keys. This does not solve off-tree strategy. [Play/replay guide](docs/play-web.md), [human benchmark guide](docs/play-web-benchmark.md).

| Model | Default / selection | Training and inference |
| --- | --- | --- |
| **v0.4.1** | Default in this checkout | #165 O, seed 2026100601, 1B nodes, opponent-sampled average; exact hash in [model card](docs/releases/v0.4.1/MODEL_CARD.md) |
| **v0.4.0** | Still selectable and downloadable | #116 R1, seed 2026093001, 100M nodes, current regret-matched policy |

For a single-model legacy launch, `--policy PATH` loads v0.4.0; `--o-candidate PATH` loads the exact O policy. Use `--data-dir` to keep separate private journals. The server binds only to loopback; see the [SSH tunnel guide](docs/play-web.md#model-and-launch).

## Limits and release plan

Each release uses fresh roots, predeclared paired gates, a direct incumbent match and independent replay of every action/settlement. Bounded LBR gives a lower bound on exploitability; turn/river weakness is conditional on fixed trees and ranges. Small opponent panels remain imprecise. The older neural Deep CFR work is not a complete Pluribus reproduction, and no six-player or 100-BB strength follows from this preview. [Research history](docs/research-history.md), [roadmap](ROADMAP.md), [release notes](docs/releases/v0.4.1/RELEASE_NOTES.md).

| Release | Game and focus |
| --- | --- |
| **v0.4.0** | Heads-up 20 BB research preview and local table |
| **v0.4.1** | Heads-up 20 BB average-policy play; prepared pending final owner publication go |
| **v0.4.x next** | #166 turn-search results, then trainer/storage options and finer abstraction |
| **v0.5** | Heads-up 100 BB and a first external benchmark |
| **v0.6 / v0.7** | Three players / four and five players |
| **v0.8** | Six players, 100 BB; confirmed profit against the scripted pool |
| **v0.9 / v1.0** | Changing lineups/stacks and full-table play / lower-end professional benchmark |

### What 1.0 means

A credible professional reference, predeclared multi-seed confirmation, legal information-safe play and reproducible evidence. Later milestones keep their own acceptance checks.

## Developers and licenses

Install `requirements-dev.txt`, then run `python -m pytest -q`. [CI](.github/workflows/tests.yml) also checks observations, sessions, arenas, solvers and recovery. Read [AGENTS.md](AGENTS.md), [rules](docs/rules.md) and the [observation contract](docs/observations.md) before changing behavior.

Repository code is [MIT](LICENSE.txt). The pinned `pokers` upstream has no verified license grant in its repository/package metadata; the owner licenses their changes to that fork under MIT, while the original authors' terms remain unresolved. [License audit](docs/releases/v0.4.0/LICENSE_AUDIT.md). Release assets contain inference only, no engine binary or private journal; reports and referenced papers retain their own rights.
