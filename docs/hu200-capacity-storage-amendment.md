# PR301 inactive model storage amendment — October 17

The owner explicitly authorized storing models that are not needed now on bb128's
2 TB drive under this PR's name, then retrieving them when needed. This expands
the earlier final-retention-only permission to inactive model storage. The owner
then explicitly authorized juggling files between the two devices so storage
does not block this PR301 campaign. This includes staging and later retrieval,
with verified copies and indexed moves, without repeated storage approval. It does
not authorize retrying the stopped campaign, running models from the HDD,
changing scientific work, or extending any deadline or resource budget.

Destination: `/mnt/data2tb/research/PR-301-hu200-capacity-cygnus/inactive-models-20261017/`,
preserving paths relative to the owned NVMe root
`/home/dberweger/hu200-capacity-20261015/`. Files remain unpacked and byte-identical.

Move only these inactive groups, approximately 56.7 GB in eleven files:

- Three restored 85M twin averages under `inputs/seed-*/final.average.gz`.
- Three fully audited 5B averages under `results/*/5000000000/`.
- Three old interrupted-attempt 10B checkpoint files under `results/*/10000000000/`,
  including seed 1302's incomplete `checkpoint.tmp`.
- Seed 1421's two interrupted export model files under the same old 10B tree.

The three **current** audited 10B recoveries and the two pending 20B checkpoint/
current/average streams under `results-recovery-20261016/` stay on NVMe. Their
receipts, original metadata and all other source/dependency files stay in place.
Unique partial evidence is preserved on the HDD, with exact restoration paths.
No PR292 files, archives, uploads or dependencies are inspected or moved.

Before copying, verify PR301 remains open, the owned jobs are stopped, and the
existing `/dev/sda1` ext4 read-write mount is actually at `/mnt/data2tb`. Reject
symlink destinations, existing transfer targets, changed or linked source files,
and insufficient destination headroom. Never create a replacement mount or write
through an absent mount onto NVMe.

Copy one file at a time, hashing the source stream. Compare intact models with
their already pinned SHA256s. Fsync each copied file, then independently reopen
and read every destination byte to verify size/SHA256. Recheck source size,
mtime, inode and device. Only after a durable verified destination and journal
entry exist may the exact original be unlinked. Preserve uncertain sources or
partial copies on any failure; do not retry automatically. Log both pre-removal
verification and completed relocation for each file.

Run the storage worker on one CPU under the existing VRM/CPU/RSS/swap/disk
watchdog, with a two-hour storage-operation ceiling inside the unchanged absolute
training deadline. One second CPU is available to the watchdog. CPU 85°C,
Thermistor 15 at 95°C continuously for 30 seconds, 512 MiB worker RSS, original
1 MiB swap baseline plus at most 256 MiB growth, and the 50 GB NVMe free floor
remain. The worker also checks the mount and destination floor during copying.
No trainer, export or audit is run. This is I/O relocation, not a scientific retry.

The manifest and journal record every source/destination, size, original mtime,
SHA256, readback result and removal. Index the receipt and retrieval instructions
in RESULTS_INDEX.md. To use a relocated model later, first admit the required
NVMe space, copy it back to its exact original path, verify its indexed size and
SHA256, and restore its recorded mtime. Do not silently replace original paths
with symlinks. The model's scientific audit status does not change by moving it.

The independent final Drive archives and final unpacked retention requirements
remain open. This temporary storage layout does not claim cloud upload or final
closeout. The previous completion ETA remains withdrawn while scientific work
is stopped; storage completion alone does not authorize a restart.

This ongoing storage permission applies only within the current PR301 campaign.
Use the HDD for inactive models, retained evidence and archive staging as needed;
keep active training/evaluation/cache work on NVMe for performance. Check current
dependencies before each move, and never relocate a file an active worker is
using. Preserve source-relative member hierarchy and update the location ledger
before the original copy is removed. Restore only the required inputs for the
next admitted stage rather than duplicating every model at once. Preserve the
50 GB NVMe floor and all other guards. PR292, other work roots and future
experiments remain outside this permission. Scientific restart after the terminal
failure still needs an explicit recovery authorization; storage approval alone
does not rerun a failed job.
