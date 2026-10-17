# Contributing

The project's goal is a strong no-limit Texas Hold'em agent that sees only what a human in its seat would see, starting heads-up and growing to six players. The [roadmap](ROADMAP.md) has the milestones and the [Help wanted](ROADMAP.md#help-wanted) list. Help is welcome at every level: a clearer doc, a new test, a faster trainer, a better web table or a research campaign.

Most useful work needs no large compute. Training the released models took billions of nodes on dedicated machines, but the engine, evaluation, web table, tools and docs can all be improved on a laptop.

## Before you start

- **Small fixes** (typos, broken links, a failing test, a clear bug): open a PR directly.
- **Anything larger:** comment on the issue or open one first, so two people don't take the same task and the approach can be agreed before you write it.
- **Research that needs more than a laptop:** open a [research proposal](#research-proposals) and wait for it to be accepted before the main run.

Read the [ground rules](ROADMAP.md#ground-rules) once. If your change touches game behavior or what a player can see, also read [docs/rules.md](docs/rules.md) and [docs/observations.md](docs/observations.md).

## Set up

You need Python 3.11, Git and a [Rust toolchain](https://rustup.rs/): installing the requirements builds the pinned [`pokers`](https://github.com/dberweger2017/pokers) engine from source. The GitHub CLI is only needed to download released models.

The checkout is large (about 270 MB, mostly recorded evidence in `docs/reports/`), and the full history is larger still. A partial clone fetches old file versions only when you need them:

```sh
git clone --filter=blob:none https://github.com/dberweger2017/CFR-NLHE-6max.git
cd CFR-NLHE-6max
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

To only play against the released models, `requirements-play.txt` is enough; the [readme](readme.md#quick-start) shows the downloads. [docs/development.md](docs/development.md) maps the code and its entry points.

## Run the checks

From the repository root:

```sh
python -m pytest -q
python -m scripts.check_game --players 6 --hands 20
python -m scripts.check_session --hands 30
```

If you change the native trainer, also run `cargo test --locked --manifest-path native/hu20-trainer/Cargo.toml --lib`.

Before committing, stage your files and run the storage check. It reads **staged** files, so run it after `git add`:

```sh
python -m scripts.check_repository_artifacts
```

CI runs the same checks on every PR. PRs that only edit Markdown get a fast prose lane; everything else runs the full suite ([docs/ci.md](docs/ci.md)). A first-time contributor's CI run may wait for the maintainer to approve it.

## Ways to contribute

**No large compute.** Docs and explanations, tests and rule regressions, the [web table](docs/play-web.md) (`apps/poker-web/`, `src/play_api/`), scripted opponents and evaluation tools (`src/arena/`), the [Slumbot client](docs/reports/slumbot-client.md), and native trainer speed (`native/hu20-trainer/`). Trainer changes must keep checkpoints byte-identical, with parity tests that show it.

**Laptop-scale experiments.** Small games such as Kuhn and Leduc, short HU20 runs, benchmarks and reproductions of published results on your hardware. Report what you find, including weak or negative results; they are kept as evidence.

**Large or paid compute.** Long training and big evaluations follow the research process below. The maintainer approves the budget and the machines before the main run. The project's machines are described in [docs/hardware-inventory.md](docs/hardware-inventory.md); design for them when your work will run there.

**Playing the bot.** Play it in the local web table, or watch two versions in the [spectator](docs/spectator.md), which shows each decision's exact probabilities. Report bugs and decisions that look wrong in an issue, with the hand history.

## Rules every change follows

- **Information:** a policy receives only its seat's legal observation: never other players' hidden cards, the deck, seeds or future outcomes. Training may use simulated payoffs as targets, never as inputs. See [docs/observations.md](docs/observations.md).
- **Rules:** cash-game no-limit Hold'em under [docs/rules.md](docs/rules.md). That file is the game contract; changes to it need the maintainer's review.
- **Strength claims:** "plays better" needs a predeclared paired comparison whose 95% interval clears zero, with regression limits written down beforehand. A finished run, a lower loss or a win against random play is useful evidence, not a strength claim.
- **No large files in Git:** models, checkpoints, replay buffers and raw traces stay in ignored local directories (`models/`, `results/`, `planning/`). CI rejects new research payloads. Put small summaries in the PR; for large evidence, say where it is and the maintainer will arrange archiving.
- **Old campaign scripts:** many files in `scripts/` launch one specific past campaign with fixed paths, hashes and budgets. Read the campaign's protocol and report before reusing one, and don't rerun a paid campaign as-is.
- **No credentials:** never commit tokens, account details or `.env` files.

## Pull requests

- One PR per coherent task, from a fork or a `feature/<short-name>` branch. Keep `main` usable.
- Commit messages describe the actual change.
- The PR description states the problem, the resulting behavior, the design choice and the evidence, in plain language. The [template](.github/pull_request_template.md) has these headings.
- Code: small modules, typed boundaries, explicit data ownership, deterministic examples and tests of real behavior. Keep logging out of game transitions and reports out of training logic.
- Comments explain decisions, constraints and non-obvious invariants; they don't narrate the code. Remove dead paths rather than keeping compatibility scaffolding.
- The maintainer reviews every external PR. Changes to rules, information access or regret math get an independent review, and an unresolved correctness question blocks training that depends on it.

## Research proposals

Open a **research proposal** issue before the main run of any experiment that needs more than a laptop, or whose result would support a claim. It declares, in advance:

- the hypothesis and the decision it informs;
- opponents, seeds and evaluation size;
- maximum cost and runtime, and the machines;
- the evaluation schedule, stopping conditions and decision rule.

Numerical failures, information leaks, broken accounting, invalid play and resource exhaustion stop a run. Weak or inconclusive results are recorded, not a reason to extend the budget. Existing protocols in `docs/` (for example the [v0.5.5 acceptance protocol](docs/hu200-v055-acceptance-protocol-v2.md)) show the format.

## Decisions kept by the maintainer

These need the maintainer's approval: changes to game scope, significant algorithm or compute tradeoffs, long or paid compute, removing valuable artifacts, releases and default-model changes, and public strength claims. Bring a recommendation, the alternatives and the evidence.

## AI coding agents

You may use them; you remain responsible for the change. The repository keeps no shared agent instruction file, so point your agent at this guide, and keep your own agent files out of your PRs.

## Licence and questions

Repository code is [MIT](LICENSE.txt), and contributions are accepted under it. The licence of the original `pokers` engine authors is still unresolved; see the [licence audit](docs/releases/v0.4.0/LICENSE_AUDIT.md).

For questions, open an issue.
