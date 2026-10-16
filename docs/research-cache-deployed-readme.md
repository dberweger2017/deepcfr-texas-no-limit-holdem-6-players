# /research: fast cache in front of Google Drive

This folder is a **fast local cache**. The long-term store is Google Drive (`deepcfr-research-results`, rclone remote `gdrive:`).

- **Files here are deleted after 7 days without use.** A PR folder is evicted when nothing in it has been read or written for 7 days.
- **Every file here must also have a copy on Google Drive.** A folder is only evicted when its receipt proves that Drive holds everything in it.

`/research` is the 1.8 TB hard disk at `/mnt/data2tb/research-cloud`. Active training, audits and caches belong on the NVMe; this disk is for results at rest. It reads and writes at about 225 MB/s, much faster than Drive's ~4 MB/s.

## For agents: saving a PR's research

1. **Keep an uncompressed copy here:** `/research/PR-<n>-<name>/`, using the same folder structure your ZIP has.
2. **Upload your ZIP to Drive from here,** which also writes the receipt:

   ```sh
   /research/.tools/research_cache.py upload PR-<n>-<name> /path/to/your-archive.zip
   ```

   This uploads to `gdrive:PR-<n>-<name>/` with `rclone copyto --immutable` (one-way: never overwrites or deletes anything on Drive). It then checks Drive's MD5 against the ZIP and each local file against its ZIP member, and records the result in `.drive-receipt.json`. Files it can't match are listed as "not yet covered", and the folder is never evicted while any exist.
3. Use `receipt` instead of `upload` for files already on Drive. `status` shows every folder's idle time and coverage.

Only bb128 uploads research files to Drive. On the Macs, the synced Google Drive folder is read-only: use it only to fetch something that isn't here, copy it out, and never delete inside it (that deletes from Drive).

## What is never evicted

- folders of **open PRs** (the open-PR list comes from GitHub; if it's unavailable, nothing is evicted that day);
- folders with a file named **`KEEP`** in them;
- folders a process has open;
- folders with **any file not covered** by a verified Drive copy, or changed since its receipt.

## Eviction

A cron job runs daily at 04:00: `research_cache.py evict`.

- **Checks first:** right before evicting, it re-reads every local file and compares it with its receipt. It reads without updating access times, so the check never counts as use.
- **Then the trash:** an evicted folder moves to `/research/.evicted/` and is deleted for good **7 days later**.
- **Log:** every decision, kept or evicted, goes to `/research/.eviction-log.jsonl`, with the Drive IDs and the restore command.

**To restore:**
- within 7 days, move the folder back from `.evicted/`;
- after that, `rclone copy gdrive:PR-<n>-<name>/ /research/PR-<n>-<name>/` and unzip the ZIP there.

## Never

- `rclone sync`, `move`, `delete`, `purge` or `dedupe` against `gdrive:`, or any `--delete-*` flag.
- Reading, printing, copying or committing `~/.config/rclone/rclone.conf`: it holds the owner's Google token.
