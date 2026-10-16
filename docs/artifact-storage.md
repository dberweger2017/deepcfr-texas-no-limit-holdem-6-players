# Git, Google Drive and the research cache

This is the owner-approved October 16, 2026 storage policy. [AGENTS.md](../AGENTS.md)
controls preservation; [RESULTS_INDEX.md](../RESULTS_INDEX.md) records actual files,
receipts, failures and retrieval provenance. Historical Mac upload receipts remain
dated evidence. **Already running PR301, PR306 and other open campaigns may finish
with their current method; do not disrupt them.** New work follows this policy.

## Locations and network

Google Drive's [deepcfr-research-results](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s)
is the long-term store for **all research files**. Everything must eventually be
represented there; bb128's 1.8 TB HDD is a disposable, faster cache in front of it.

| Location | Purpose |
|---|---|
| bb128 `/research/PR-<n>-<name>/` | Uncompressed research files at rest, in the same structure as ZIP members. `/research` is the owner's symlink to `/mnt/data2tb/research-cloud`; use the real path until the symlink exists. |
| `gdrive:PR-<n>-<name>/` | The same existing per-PR Drive folder; `gdrive:` is already rooted at deepcfr-research-results. All new research uploads originate from the bb128 cache. |
| M1/M4 `~/Local/Research-Cloud/` | **Read-only** synced Drive access for retrieval. Fetch missing material, copy it into an ignored working directory, and never write/delete inside the synced folder. A deletion there deletes from Drive. |
| Authorized host's NVMe / Mac ignored `models/`, `results/`, `planning/` | Active training, audits, derived caches and restored working inputs. No heavy live work on the HDD or streamed folder. |

Use the exact existing per-PR name across cache and Drive. The HDD is about
225 MB/s against Drive's roughly 4 MB/s. bb128 currently reaches the router by
Wi-Fi. The M1 is often away and reaches bb128 through a roughly 1.2 MB/s Tailscale
relay, so it can keep fetching from Drive when needed. Prefer restoring through
bb128 so the cache fills. Only one large upload runs at a time on the shared
uplink, without competition with Slumbot play.

The owner/Claude are arranging a private bb128 ↔ M4 Ethernet cable:
`192.168.77.1` / `192.168.77.2`, no internet route, about 110 MB/s expected.
**Do not rely on it until confirmed working.** This PR does not configure it or
claim a throughput test. A missing disk, symlink, authentication or network
prerequisite must be described in the PR for the owner; retain originals and
leave the blocked work pending rather than claiming completion.

## Save research through bb128

1. Check the owning PR's current status and authorization. Preserve an
   **uncompressed** final copy in `/research/PR-<n>-<name>/` with the same paths
   as the ZIP. Include raw results, checkpoints/exports, failures, partials,
   source/config/input provenance and manifests. Exclude all credentials.
2. Build the ZIP with embedded `ARCHIVE-MANIFEST.json`, local member readback,
   archive bytes/SHA256 and manifest SHA256. Keep the final files at rest in
   the cache. For Mac work, copy the files/archive to bb128's cache rather
   than staging a new upload in the Macs' synced Drive folder.
3. Use the already deployed tool on bb128:

   ```sh
   /research/.tools/research_cache.py upload 'PR-<n>-<name>' '/research/PR-<n>-<name>/evidence.zip'
   # If /research is not yet present, use /mnt/data2tb/research-cloud instead.
   /research/.tools/research_cache.py receipt 'PR-<n>-<name>' /path/to/already-uploaded.zip
   /research/.tools/research_cache.py status
   ```

   `upload` uses one-way `rclone copyto --immutable`. `receipt` records files
   already on Drive. Both compare the Drive object's MD5/size with the ZIP,
   and each uncompressed local file with its ZIP member's size/CRC32, then
   write `.drive-receipt.json`. An uncovered or changed file blocks eviction;
   an upload alone is not complete coverage of the uncompressed folder.
4. Record in RESULTS_INDEX: owning PR/source/run identity; Drive file ID/path,
   MD5, archive bytes/SHA256, embedded manifest path/SHA256 and local readback;
   receipt/coverage status; required model/input member paths and SHA256s;
   every failure/partial and **both retrieval paths**. A compact linked receipt
   can carry details. Drive MD5 upload acceptance complements archive/member
   SHA256 provenance; do not call it a remote SHA256/member audit.

`/research/.tools/research_cache.py` is the deployed implementation. The
[repository tool](../scripts/research_cache.py) was imported byte-for-byte;
[import hashes and review findings](artifacts/research-cache-import-20261016.md)
distinguish subsequent repository fixes from deployed behavior. **Deletion
logic fixes require owner review and deployment approval.** Do not replace the
live tool, change its cron entry, or run live eviction as part of this PR.
The audit found deployed gaps in process-error handling, PR pagination and
purge protections; the linked report identifies the fixes still awaiting review.
The [copied deployed README](research-cache-deployed-readme.md) records the live
usage/policy; no deployed README or tool was modified by this import.

## Retrieve, trash and restore

While cached, use `/research/PR-<n>-<name>/` as the fast source and copy needed
files into a fresh ignored working directory. Always retain this Drive fallback:

```sh
rclone copy 'gdrive:PR-<n>-<name>/' '/research/PR-<n>-<name>/'
```

After a miss/eviction, restore the archive through bb128, verify its indexed
SHA256 and manifest, extract the named files into the same uncompressed cache
structure, and verify required member SHA256s before research use. Restore or
rebuild `.drive-receipt.json` with `receipt` before considering the cache covered.
Never overwrite an active input. On a Mac that cannot efficiently reach bb128,
fetch through its read-only synced Drive folder and copy the file out. Released
models remain available through pinned release downloads; no research binaries
enter Git without an exact-path, explicitly owner-approved release exception.

The owner explicitly authorized **one scheduled cleanup exception** on October 16:
the already installed bb128 user cron runs `research_cache.py evict` daily at
**04:00 in the host's cron timezone**. It applies only to this cache, not Mac
working roots, NVMe originals, other agents' inputs, shared Git or general cleanup.
A PR folder can be evicted only when all these conditions hold:

- No research file has been read or written for seven days. Receipt activity
  and folder listings do not count; verification reads use Linux `O_NOATIME`.
- Every local file has receipt coverage and unchanged size/mtime; each covered
  Drive object still has the same size/MD5. Immediately before eviction, every
  local file is reread and matches its recorded content checksum.
- It is not an open PR's folder. If GitHub's open-PR list is unavailable, the
  entire eviction/purge run stops without deleting anything.
- There is no `KEEP` marker and no process has the folder open. Missing or
  incomplete inspection must retain data; never infer safety from an error.

Eviction first moves the folder to `/research/.evicted/` with an eviction-time
stamp in its name. Permanent deletion is allowed only after **another seven
days measured from that stamp**, never from the directory's old mtime. Before
that, restore locally by moving the held folder back to its original cache
path after checking it will not overwrite anything. Afterward use the Drive
command above. Decisions go to `/research/.eviction-log.jsonl`, including
Drive IDs and restore commands where receipt information is available.
Repository fixes additionally recheck all protections before permanent purge;
they are **not deployed** until the owner approves the reviewed change.

## Boundaries and credentials

Never read, print, copy, archive, commit or transmit
`~/.config/rclone/rclone.conf` (the owner's Google OAuth token). Agents never
create/change Google credentials or sharing. Never use `rclone sync`, `move`,
`delete`, `purge`, `dedupe`, any `--delete-*` flag, or overwrite/remove existing
files **against gdrive:**. Local cache trash/purge is separate from Drive;
no tool operation is allowed to delete or change Drive objects.

Source, small configurations/fixtures, readable reports and compact receipts
belong in Git. Training checkpoints, inference exports, replay buffers, bucket
tables, raw traces, bulky outputs, complete logs and generated research bundles
belong in Drive, with at-rest cache copies. A filename such as `manifest.json`
does not make bulky payload data a compact Git receipt.

## Repository guard

`python -m scripts.check_repository_artifacts` checks the staged index;
`--revision HEAD` checks a commit. It rejects new model/archive/database formats,
raw JSONL traces, tracked files under local research roots, and any blob larger
than **1 MiB**, including large plain JSON. Case variants and interrupted suffixes
are checked. Small JSON summaries, CSV tables, report images and source remain
eligible. This is a maintenance rule, not a content classifier: authors still
must follow the storage contract for data that passes the mechanical check.

[The exact-path retention list](../configs/repository-artifacts.json) pins existing
legacy payloads by Git blob ID and byte count. It is **not** a Drive manifest or a
list of files approved for deletion. Existing payloads remain to avoid breaking
replay, tests or active research. Changed bytes or copies at new paths fail.
Do not regenerate the retention list to admit new experiment output. A necessary
fixture exception needs a specific reviewable reason and exact bytes; a release
model exception additionally requires explicit owner approval and an exact-path
ignore exception, as AGENTS.md requires.

`.gitignore` prevents common accidental staging; CI also checks objects already
staged or force-added. Git blob IDs identify retained repository bytes and must
never be substituted for the archive/member SHA256s required for restoration.

## Originals outside the cache

The scheduled exception does not relax NVMe/Mac original-retention rules.
Before copying, archiving, moving or cleaning research, check the owning PR's
current status. Protect open PR roots/dependencies unless the owner specifically
authorizes that action. Retire merged originals only after current matching
upload acceptance and dependency review. Trust confirmed Drive uploads and
existing manifests/provenance for cleanup; do not redownload archives or repeat
payload/member hash audits merely to remove originals. Match selected paths,
sizes and modification history; retain changed or uncertain files. Record exact
removed paths and restoration sources in a cleanup receipt and RESULTS_INDEX.
Never delete inside a Mac synced folder or force offloading.

The October 14 standing authorization for inactive rebuildable caches/output
still requires no data loss, no active dependency, and recorded source/input
pins plus regeneration commands. Unique scientific evidence is not a cache
merely because it could be recomputed. Code retirement remains a separate
caller/fixture/source and regression review. The only unattended cleanup
exception is the owner's existing bb128 cache job; this policy schedules no
other work and authorizes no new scheduler or live deployment.
