# Uniform-random HU20 calibration

This separate control samples uniformly over the exact concrete menu returned
by `choices(view, raise_cap=None, free_fold=False)`. It uses the existing native
HU20 engine, 2,000-chip stacks, alternating positions, session journal and
persisted private per-session bot RNG. It does not load or change B100M, choose
arbitrary native sizes, translate actions or inspect hidden opponent cards.

Run from this PR's source checkout and the existing source-install environment:

```sh
python -m src.play_api.server --uniform-random \
  --data-dir results/luna-random --port 8765 --source-version <exact-source-sha>
```

Open `http://127.0.0.1:8765/`, enter the token from the private data directory,
and create a **restricted benchmark with target 100**. The server rejects casual
or free-sizing sessions for this control. Host/Origin/token/asset restrictions
and benchmark-safe visibility are unchanged. No account, public hosting or
external API key is needed. The token is never put in a URL or provenance.

The UI and export identify **Uniform restricted random**, adapter
`uniform-restricted-v1`, format `builtin-uniform-restricted-v1`. Its SHA-256
identifies the canonical JSON `DEFINITION` in
[`uniform_random.py`](../src/play_api/uniform_random.py), not a trained artifact.
Record the source SHA separately. Trained/fallback lookup counts are inapplicable;
random decisions are not blueprint fallback.

Dispatch a fresh `gpt-6-luna` / `high` child with no primary conversation history,
using the unchanged frozen player prompt and browser-only harness. The player
operates only rendered controls. Preserve technical recoveries; no scores or
poker advice may be supplied. The 100-hand result is a weak-control calibration,
reported separately from the shortened B100M primary. Do not extend it without
approval.

After completion, fetch the actual HTTP benchmark export, run the reporting
helper, and replay the private journal with the control definition hash:

```sh
python -m scripts.report_luna_browser PRIVATE_DB ROLLOUT TOKEN_FILE OUTPUT_DIR
python -m tests.play_ui.verify_records PRIVATE_DB --expected-hash <definition-sha256>
```

Private journals, tokens and full transcripts remain outside Git. Publish only
the sanitized export, decision/hand tables, observable metadata/audit and resource
samples. Stop the owned service after export and validation. The UI's progress
now uses the server's hand number, so a finished hand and its next-hand button
no longer display the same ordinal. This narrow presentation correction does
not change any poker action or settlement.
