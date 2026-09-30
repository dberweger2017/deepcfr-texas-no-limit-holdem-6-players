# Continuous integration

I keep the existing final `test` status for pull requests and main. The workflow always runs;
it does not use workflow-level path filters that could leave a required check
pending. Its summary states whether it selected full validation or prose checks.

## What runs

| Event/change | Validation |
| --- | --- |
| PR adding/editing only ordinary tracked `.md` files | CI routing regressions and UTF-8 checks of the changed tracked prose |
| PR changing `docs/rules.md` | Full validation: this is the game contract |
| PR changing code, workflows, dependencies, configs, prompts, manifests, fixtures or result data | Full validation |
| PR deleting/renaming files, changing file types, or adding symlinks/executable files | Full validation |
| Missing/unavailable comparison or unrecognized change | Full validation |
| Push to main or manual workflow dispatch | Full validation, even for prose |

The prose path does not install Torch or build the native engine. It checks
encoding, not factual accuracy, links or whether a rewritten README command is
correct; those remain review responsibilities. A manual run of **Tests** always
forces the complete existing suite. No test is deleted, and no inference or
training behavior changes.

PR classification compares the event's base revision with the checked-out merge
snapshot, not just the latest commit. A prose follow-up to a code PR therefore
cannot skip its code tests. Checkout includes two levels of history; an
unavailable base safely falls back to full validation. The classifier reads Git
blobs and modes, not ignored files or filesystem symlink targets.

## Full validation in two runners

The `scope` job classifies the complete change and tests the classifier. Code
changes start two `full` jobs on separate runners. A small pytest plugin assigns
every collected test to exactly one shard by hashing its repository-relative
file path. All parametrizations and tests in a file stay together, preserving
collection order. New test files join automatically; there is no manual list
that can omit future tests. Two measured exceptions place the intact
`test_hu20_scaling.py` and `test_hu20_scaling_recovery.py` files on shard 1:
their largest cases alone took 93.73s and 59.70s in the first run, while the
default split took 480.16s versus 194.05s. These are assignment hints, not
selection/exclusion lists; every collected test still belongs to one shard.
Separate runners avoid shared output-file races.

The engine, session, arena, solver, neural baseline and recovery smoke/replay
commands also run once, on shard 0. Neither shard cancels the other on failure.
The final `test` job accepts full validation only when both shards and those
commands pass. A skipped full job is accepted only after successful prose
classification/checks. Failed, cancelled or missing checks cannot produce a
green final status. Ordinary local `pytest` still runs the entire suite.

This trades a second dependency installation and runner allocation for shorter
elapsed time. It does not reduce test coverage or total compute. Runner queues
and uneven file costs can limit the speedup; duration output makes that visible.

## Superseded work

A new commit cancels older **Tests** runs for the same PR. Different PRs have
different concurrency groups, so agents do not cancel each other's checks.
Main has its own group: a newer main commit supersedes the old main run, but
**every main run is full**, retaining coverage of accumulated code changes.
Manual runs on a branch share that branch's group and also run full validation.

Do not cancel another PR merely to free a runner, and do not reuse an old green
check for a changed integration. A new main merge can combine changes absent
from an earlier PR test, so its final full check still matters.

## Finding expensive tests

Full runs print the 25 slowest test phases taking at least one second:

```sh
python -m pytest -q --durations=25 --durations-min=1
```

This adds measurements without rerunning tests. The existing engine, session,
arena, solver, neural baseline and recovery smoke/reproduction commands remain.

On September 30, the corrected analysis PR spent about 7m30s in pytest versus
25s on dependency installation. Its combined benchmark PR spent 12m12s in
pytest. The 39 focused play checks took 3.36s locally. These are different source
snapshots/platform measurements, not a controlled speed comparison. They show
why deleting many cheap tests or tuning installation first would miss most of
the cost. New timing output will identify individual slow tests before changing
fixture budgets. Two file shards reduce sequential waiting while preserving
those tests; their actual elapsed time must be measured on GitHub.

The first run of [PR #131](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/131)
passed all 1,031 tests and existing smoke/recovery gates. Its pytest shards took
8m00s and 3m14s; the whole workflow took 9m27s. The preceding main run spent
12m08s in pytest alone. This is observed integration timing on separate runners,
not a controlled benchmark or a promise for future queue/run times. That first
run motivated the two file-assignment hints above; its timings precede them.

To reproduce either shard locally:

```sh
python -m pytest -p scripts.ci_shard --ci-shard 0 --ci-shards 2 -q --durations=25 --durations-min=1
python -m pytest -p scripts.ci_shard --ci-shard 1 --ci-shards 2 -q --durations=25 --durations-min=1
```

Validate routing without installing poker dependencies:

```sh
python -m unittest tests.test_ci_scope -v
```

The tests exercise real temporary Git histories, merge snapshots, unavailable
bases, shallow merge checkouts, code/data/contracts, removals, renames,
executable modes, symlinks, and tracked-blob reads. Shard regression tests run
actual pytest collection/execution in isolated fixtures, verifying disjoint,
complete parametrized coverage and rejection of invalid shard settings.
The final workflow shell gate is exercised across all 48 combinations of
successful/failed/cancelled/skipped jobs and full/prose/missing routing output.
The first CI-speed PR itself changes workflow/code, so it receives full CI;
future eligible prose-only PRs use the fast path. No measured GitHub fast-path
latency is claimed until such a PR runs it.
