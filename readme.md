# A local heads-up poker research bot

This bot plays heads-up (HU, one-on-one) no-limit Texas hold'em. Each model learns from billions of simulated self-play decisions with counterfactual regret minimization (CFR), the family of methods behind Libratus and Pluribus, on a simplified game: similar hands share card buckets and bets come from a fixed menu of sizes. You can play against it, or watch two versions play each other, in a local web app. The repository name is historical: the project began as a six-player Deep CFR experiment, and six players remain a later roadmap goal (v0.8).

There are two release lines, HU20 and HU100, for 20 and 100 big-blind stacks, with no rake or ante and fresh stacks each hand. Both play tabular, linearly weighted, opponent-sampled CFR averages.

**v0.5.3, heads-up 100 BB (newest).**
- **The model:** a 5B-node average (seed 2026100601) trained on a larger betting menu (min, half pot, pot, 1.5× pot, 2× pot and all-in) with finer postflop card buckets, plus translation for off-menu bet sizes.
- **Head to head:** it beats v0.5.2 by **+37.10 [30.16, 44.04] BB/100**, with translation on for both. All three training seeds beat v0.5.2: +47.0, +41.0 and +45.4 (#296).
- **Scripted opponents**, BB/100: check/call +269, random +107, tight-aggressive +37, loose-aggressive +35, pot-size pressure +27, matched-pot pressure +20, half-pot pressure +16. Pot-size pressure, which beat v0.5.2, is now beaten.
- **Limits:** no external benchmark has been played; against matched-pot pressure it scores less than v0.5.2 did (−26.6 [−40.8, −12.3] on the same deals).
- The download is 4.6 GB in twelve assets; the model reassembles itself on first start. [Model card](docs/releases/v0.5.3/MODEL_CARD.md) · [install and run](docs/releases/v0.5.3/INSTALL.md)

**v0.5.2, v0.5.1 and v0.5.0, heads-up 100 BB.** The 4B-, 2B- and 1B-node averages on the original menu, still available. [v0.5.2 model card](docs/releases/v0.5.2/MODEL_CARD.md) · [v0.5.1](docs/releases/v0.5.1/MODEL_CARD.md) · [v0.5.0](docs/releases/v0.5.0/MODEL_CARD.md)

**v0.4.2, heads-up 20 BB.**
- **The model:** a 10B-node average, fixed seed 2026100601. It is the default on the 20 BB table; v0.4.0 and v0.4.1 remain selectable.
- **Results:** the three retained 10B lineages beat matched-seed 1B v0.4.1 directly by **+3.50 [+1.63, +5.37] BB/100**, and all declared candidate checks pass.
  - Fresh aggregate bounded LBR is **−1.092 [−4.965, +2.782] BB/100**. That narrowly clears the unchanged lower > −5 safeguard, with half-width **3.874**.
  - Its point estimate favours 1B on LBR, so this doesn't establish an LBR improvement or each seed's non-regression.
- **What the intervals cover:** they're nominal paired 95% intervals, conditional on the saved lineages. No full-game exploitability certificate or human or professional strength follows.
- [Updated decision table and limits](docs/reports/hu20-v042-lbr-confirmation.md) · [v0.4.2 model card](docs/releases/v0.4.2/MODEL_CARD.md) · [earlier v0.4.0 → v0.4.1 comparison](docs/reports/v0.4.1-release.md)

## Watch two bots

In the local web app, choose **Bot-vs-bot spectator**, select two pinned models, then use **Play**, **Pause** or **Step**. Each decision shows its exact policy probabilities, selected action, lookup status and acting bot's legal perspective. Sessions retain both release/model/manifest identities and replayable hand history; v0.4.0 and v0.4.1 remain selectable. See the [spectator guide and independent audit command](docs/spectator.md).

## Quick start

Use Python 3.11, Git and the GitHub CLI. From a fresh checkout:

```sh
git clone https://github.com/dberweger2017/CFR-NLHE-6max.git
cd CFR-NLHE-6max
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-play.txt
mkdir models
gh release download v0.4.2 --pattern 'O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz' --dir models
gh release download v0.4.1 --pattern 'O1B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz' --dir models
gh release download v0.4.0 --pattern 'B100M-HU20-current-seed-2026093001.json.gz' --dir models
python -m scripts.verify_v042_model models/O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz
python -m scripts.verify_v041_model models/O1B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz
python -m scripts.verify_v04_model models/B100M-HU20-current-seed-2026093001.json.gz
python -m src.play_api.server --models-dir models
```

`mkdir models` and downloads fail rather than overwrite existing artifacts. If the directory exists, inspect it or use a fresh destination. Installation builds the pinned native `pokers` engine; follow its build error instructions if your platform needs a Rust toolchain.

Open **http://127.0.0.1:8765/** and enter the token from `results/play-web/access.token` locally. Keep tokens out of URLs and screenshots. Choose **v0.4.2** (default), **v0.4.1** or **v0.4.0**, choose bet sizing, create a session and deal. Seats alternate each hand. Model choice is frozen for that session; refresh resumes it, including after server restart. Use a new session to change models. All three pinned files are required for model selection; a missing or corrupt default causes an error, never a silent replacement. The average readers use compact storage; the M4 runtime/replay smoke peaked at 2.76 GiB with all three releases. Allow at least 3 GiB of RAM.

Restricted mode uses the concrete training menu. Free sizing executes any native-legal integer-chip human wager exactly; the bot keeps its trained menu and unchanged uniform fallback on missing/zero-mass keys. This does not solve off-tree strategy. [Play/replay guide](docs/play-web.md), [human benchmark guide](docs/play-web-benchmark.md).

| Model | Default / selection | Training and inference |
| --- | --- | --- |
| **v0.4.2** | Default in this checkout | #185 O, seed 2026100601, 10B nodes, opponent-sampled average; exact hash in [model card](docs/releases/v0.4.2/MODEL_CARD.md) |
| **v0.4.1** | Still selectable and downloadable | #165 O, seed 2026100601, 1B nodes, opponent-sampled average; exact hash in [model card](docs/releases/v0.4.1/MODEL_CARD.md) |
| **v0.4.0** | Still selectable and downloadable | #116 R1, seed 2026093001, 100M nodes, current regret-matched policy |

For a single-model legacy launch, `--policy PATH` loads v0.4.0; `--o-candidate PATH` loads the exact v0.4.1 O1B policy (unchanged legacy option). Use `--data-dir` to keep separate private journals. The server binds only to loopback; see the [SSH tunnel guide](docs/play-web.md#model-and-launch).

## Limits and release plan

Each release uses fresh roots, predeclared paired gates, a direct incumbent match and independent replay of every action/settlement. Bounded LBR gives a lower bound on exploitability; turn/river weakness is conditional on fixed trees and ranges. Small opponent panels remain imprecise. The older neural Deep CFR work is not a complete Pluribus reproduction, and no six-player or 100-BB strength follows from this preview. [Research history](docs/research-history.md), [roadmap](ROADMAP.md), [release notes](docs/releases/v0.4.2/RELEASE_NOTES.md).

| Release | Game and focus |
| --- | --- |
| **v0.4.0** | Heads-up 20 BB research preview and local table |
| **v0.4.1** | Heads-up 20 BB average-policy play; [release and verified assets](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.4.1) |
| **v0.4.2** | Heads-up 20 BB 10B-node average; [stable release and verified assets](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.4.2) |
| **v0.4.x next** | #166 turn-search results, then trainer/storage options and finer abstraction |
| **v0.5.0** | Heads-up 100 BB 1B-node average with translation; internal and scripted evaluation only, as no suitable free public 100 BB opponent has been found; [release and verified assets](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.5.0) |
| **v0.5.1** | Heads-up 100 BB 2B-node average with translation; beats v0.5.0 head to head by +29.5 BB/100; [release and verified assets](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.5.1) |
| **v0.5.2** | Heads-up 100 BB 4B-node average with translation; beats v0.5.1 head to head by +17.9 BB/100, repeated on two other seeds; [release and verified assets](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.5.2) |
| **v0.5.3** | Heads-up 100 BB 5B-node average on the larger menu with translation; beats v0.5.2 head to head by +37.1 BB/100, all three seeds beat v0.5.2; [release and verified assets](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.5.3) |
| **v0.5.5** | Heads-up 200 BB, trained at that depth and benchmarked against Slumbot; results and uncertainty published |
| **v0.6 / v0.7** | Three players / four and five players |
| **v0.8** | Six players, 100 BB; confirmed profit against the scripted pool |
| **v0.9 / v1.0** | Changing lineups/stacks and full-table play / lower-end professional benchmark |

### What 1.0 means

A credible professional reference, predeclared multi-seed confirmation, legal information-safe play and reproducible evidence. Later milestones keep their own acceptance checks.

## Developers and licenses

[Repository map and commands](docs/development.md), [artifact storage](docs/artifact-storage.md),
and [maintenance audit](docs/reports/necessary-cleaning.md).

The play-only installation excludes neural research dependencies. For the full
research and test environment, install `requirements-dev.txt`, then run `python -m pytest -q`. [CI](.github/workflows/tests.yml) also checks observations, sessions, arenas, solvers and recovery. Read [AGENTS.md](AGENTS.md), [rules](docs/rules.md) and the [observation contract](docs/observations.md) before changing behavior.

Repository code is [MIT](LICENSE.txt). The pinned `pokers` upstream has no verified license grant in its repository/package metadata; the owner licenses their changes to that fork under MIT, while the original authors' terms remain unresolved. [License audit](docs/releases/v0.4.0/LICENSE_AUDIT.md). Release assets contain inference only, no engine binary or private journal; reports and referenced papers retain their own rights.
