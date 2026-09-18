# Runpod multi-street operations

`scripts/runpod_multistreet.py` supervises scientific workers inside a
disposable pod. It owns process groups, deadlines, thread limits, host
provenance, and durable status. The scientific runner owns plans, seeds,
reference caches, fitting, checkpoints, and scientific reports.

The wrapper never contacts Runpod and never reads provider credentials. A
separate provider watchdog must be armed after provisioning and must terminate
the specific pod independently of SSH. Do not put pod IDs, API keys, or private
keys in the repository or the campaign archive.

## Runner contract

The command after `--` is an argv template, not a shell string. Supported
placeholders are `{worker}`, `{seed}`, and `{out}`. The stratified campaign
runner should accept the following stable interface:

```text
python -m scripts.stratified_campaign \
  --plan PLAN.json --seed SEED --out OUTPUT [--resume]
```

It should publish its own `status.json` and final `report.json` below the
worker output. The wrapper records the exact expanded argv and does not import
or infer scientific settings.

## Remote setup and launch

Use a reviewed commit and a fresh disposable pod. The known-good CPU setup is
Python 3.11, CPU PyTorch 2.5.1, and the repository's pinned requirements:

```bash
export UV_CACHE_DIR=/workspace/uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/python
export CARGO_HOME=/workspace/cargo
export RUSTUP_HOME=/workspace/rustup
export PATH="$CARGO_HOME/bin:/root/.local/bin:$PATH"
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
cd /workspace
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git poker
cd poker
git checkout REVIEWED_COMMIT
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest tests/test_runpod_multistreet.py -q
```

For the 32-vCPU/64-GB candidate, run three independent seed workers with one
thread each. A seven-hour wrapper envelope leaves one hour for retrieval; an
independent eight-hour provider cutoff leaves additional shutdown headroom.
Recalculate these values from the final quote before provisioning.

```bash
cd /workspace/poker
.venv/bin/python -m scripts.runpod_multistreet run \
  --out results/runpod-multistreet-ops \
  --seeds 911,919,929 \
  --workers 3 \
  --max-runtime-seconds 25200 \
  --retrieval-reserve-seconds 3600 \
  -- python -m scripts.stratified_campaign \
    --plan configs/holdem/multistreet-campaign.json \
    --seed '{seed}' --out '{out}'
```

The shell only supplies the argv template; the wrapper launches the child
without a shell. If the scientific runner stops partway through, rerun the
same command with `--resume` after checking its own status files. Completed
workers with exit code zero are skipped; failed or incomplete workers are
started again in their existing declared output location only if the runner's
resume contract permits it.

## Status and exit behavior

```bash
.venv/bin/python -m scripts.runpod_multistreet status \
  --out results/runpod-multistreet-ops
```

The wrapper writes `ops-status.json`, `host-provenance.json`, one
`worker-NNN.json`, one `worker-NNN.log`, and `retrieval-ready.json`. Worker
records include the expanded argv, PID, start/finish times, and actual process
return code. Child process groups receive `SIGTERM` at the work deadline and
`SIGKILL` only after the configured grace period.

The wrapper returns 0 only when all workers exit 0. It returns 1 for a worker
failure, 124 when the work deadline is reached, and 130 for an operator
interrupt. All terminal states set `retrieval_ready: true`, including partial
or failed runs, so a failed campaign's evidence is not silently lost.

## Retrieval and shutdown

When `retrieval-ready.json` is true:

1. Record the wrapper status and inspect every worker report and exit code.
2. Archive the complete operations directory, scientific outputs, logs,
   manifests, caches, and checkpoints on the pod. Write the archive SHA-256.
3. Copy to a local `.partial` path, rename only after transfer completes, and
   verify the archive SHA-256 locally. Keep the original archive and all failed
   worker outputs.
4. Request provider termination through the already-armed provider control or
   the signed-in console. Confirm compute and temporary storage show `$0/hr`.
5. Delete the network volume only after local verification and after confirming
   that the archive is readable. Record the provider's final status and the
   observed storage cleanup.

SSH loss, a Python exception, and a scientific worker timeout must not be the
only shutdown mechanism. The provider watchdog is external to this wrapper and
must have a deadline later than the wrapper's retrieval reserve but earlier
than the campaign spending cap.
