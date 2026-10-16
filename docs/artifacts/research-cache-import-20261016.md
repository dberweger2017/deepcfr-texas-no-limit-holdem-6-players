# Research cache tool import — October 16, 2026

Imported over SSH from bb128, without reading or copying rclone credentials:

| Deployed source | SHA256 at import |
|---|---|
| `/mnt/data2tb/research-cloud/.tools/research_cache.py` | `d03af1d5b0d3161723d44cbadf57cee62f5df60baaf3bd7b1fc32e340d49dcce` |
| `/mnt/data2tb/research-cloud/README.md` | `21363a9bd97e425343b19d9c72e2e4cf3a444be313173ecbb53783c84934a049` |

The initial import commit `e2a01657` contains a byte-identical `scripts/research_cache.py`
and `docs/research-cache-deployed-readme.md`. Later repository fixes are separate
commits; this source hash remains the deployed baseline, not a claim that a
patched repository copy is already deployed.

Read-only activation observations: `/research` is the owner's root-owned symlink
to `/mnt/data2tb/research-cloud`; the deployed tool is executable. The owner's
user crontab contains:

```text
0 4 * * * /usr/bin/python3 /mnt/data2tb/research-cloud/.tools/research_cache.py evict >> /mnt/data2tb/research-cloud/.tools/cron.log 2>&1
```

No deployment, cron/configuration/credential change, live upload or eviction is
part of the import. Eviction changes in this PR require owner review before
any deployment. Existing PR301/306 and other running campaigns keep their
current archival method until they finish. The private M4 cable is not relied on.

## Repository audit and fixes

The patched repository script's SHA256 is
`a4546d6559979fabbc2879d485ce3a312e5d63c3db0c96c79fce1e3cc41aa9ac`.
A final read-only SSH check confirmed the deployed script and README still have
the baseline hashes above. The copied README remains byte-identical; the main
[storage contract](../artifact-storage.md) gives the full owner-approved policy
and distinguishes proposed safeguards from the live implementation.

| Finding in the imported tool | Repository change |
|---|---|
| `lsof` errors or warnings could be treated as no open files. | Only an empty exit 1 proves no matches; errors retain the folder. Recheck after content verification. |
| Only the first 100 open PRs were fetched. | Read all GitHub pages; unavailable or invalid lists abort eviction and purge. |
| Permanent purge checked elapsed trash time without rechecking preservation conditions. | Recheck open PR, KEEP, process, recent use, receipt coverage, live Drive and local content before purge. |
| Re-receipting a replaced ZIP retained coverage for members no longer in it. Covering-object references were not validated. | Revoke old coverage for that object, require a recorded covering object, and reject duplicate normalized ZIP member paths. |
| Symlinked entries could be skipped; archive/receipt paths, trash and audit-log destinations lacked boundary checks. | Refuse symlinks/nonregular entries and escaping member paths; require an ordinary local trash directory and writable regular audit log before mutation. |
| Folder mtimes counted as research use; platforms without `O_NOATIME` silently used ordinary reads. | Count only research-file reads/writes, require `O_NOATIME`, and recheck inventory, metadata and activity after content verification. |
| Kept/purged decisions did not consistently include available Drive IDs and restoration details. | Include receipt-derived IDs and restore instructions across normal decision records. |

The deployed version **already used the eviction name stamp** for the seven-day
trash interval. This PR preserves that fix and tests it explicitly, including a
moved folder with a 100-day-old mtime, just before seven days and at seven days.
It does not claim to have discovered or repaired that already-fixed bug.

## Validation

```sh
python3.11 -m unittest tests.test_research_cache -v
```

The final suite ran **31 tests: 30 passed and one Linux-only test skipped on the
Mac coordinator**. Linux CI must exercise the real `O_NOATIME` flag and verify
unchanged nanosecond atimes; Mac lifecycle tests emulate that read behavior only
inside their temporary folder. A separate test ensures unsupported platforms
fail closed. CI is required before the PR is ready for owner review.

All lifecycle fixtures use temporary local cache/Drive folders, executable fake
rclone/lsof commands and mocked GitHub responses. No test contacts Drive,
GitHub or bb128. The fake rclone rejects every command except immutable
`copyto` and read-only `lsjson`; the full upload/receipt/status/evict/purge
lifecycle also asserts those are the only commands issued, with no deletion
flags. Tests cover missing/changed/uncovered content, KEEP, open PRs and GitHub
failure, process inspection failure, receipt coverage, access times, path
boundaries, trash timing and protections after the initial move.

For comparison, running the first 25 regression cases against the exact
imported script produced **13 failures, zero errors and one Linux-only skip**.
The added boundary/coverage cases bring the final suite to 31. Tests retain
changed content even if its size and mtime are preserved; matching metadata
alone is never proof before deletion.

The source import and offline fixes are complete; **deployment remains an owner
decision after review of this concrete diff**. The live cron continues using
the imported baseline, including the defects documented above. No new job was
scheduled, no live cache folder was moved or removed, and no credentials,
Google sharing or active research routes were changed.
