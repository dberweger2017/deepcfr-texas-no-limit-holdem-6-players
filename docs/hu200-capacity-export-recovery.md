# PR301 export recovery after the second storage stop

The owner explicitly authorizes returning, correcting the ETA for downtime and
moving inactive PR301 files to the2TB drive to prevent another avoidable disk
stop. This grants one recovery of the unfinished1302 export/audit, without
repeating training or1421/1422 completed audits. It does not authorize future
scientific retries, experiments, sample reductions or changed trainer code.

Before resuming, relocate all inactive owned research files of at least100MB,
except the active1302 20B checkpoint, from the three owned results trees to
`/mnt/data2tb/research/PR-301-hu200-capacity-cygnus/inactive-before-export-recovery-20261017/`.
Preserve original relative paths, known pinned SHA256s, source identity/mtime,
every partial, and a durable journal. Each source is removed only after fsync
and independent full destination SHA256 readback. Shared Git/source/dependencies,
other PRs, uploads and synced folders remain untouched. Smaller unique evidence
stays on NVMe. A missing/wrong mount, hash mismatch or storage stop preserves
source/partial files and fails closed. Storage setup failures remain recorded.

The measured selection is21 files/70,431,280,579 bytes. Final admission requires
the completed relocation receipt, matching HDD metadata, all prior clock and
failure pins, byte-identical1421/1422 audit receipts and1302 saved-byte receipt,
fresh1302 checkpoint/binary/#163 hashes, original memory/swap/thermal admission,
and **at least93GB physical free**:23GB full export/audit output,50GB floor and
an additional20GB physical growth buffer. This strengthens startup admission;
the unchanged global watchdog still enforces every original guard throughout.
Unbounded writes outside the owned root cannot be controlled or reclaimed by
this PR; such writes still trigger the protective stop. Never relax the floor.

Use a fresh immutable orchestration source, phase`export-recovery-20261017`,
output`results-export-recovery-20261017/2026101302/20000000000/`, guard and stop
files. The existing audit worker reads the unchanged20B checkpoint at its old
path, independently verifies all saved bytes/rows, invokes the same qualified
native exporter and audits every current/average row. Its unchanged native
record is copied as metadata; no checkpoint/model bytes are copied to new
NVMe paths. One512MiB streaming helper uses one physical CPU; total<=8. After
success, verified checkpoint/current files move to HDD, leaving the average
on NVMe. Publish all-three completion only after this audit and storage pass.

Keep every original clock and prior ledger byte-identical. Seal a new immutable
ledger at successful recovery admission: add only the interval from the recorded
failure**2026-10-17T14:25:09.233853UTC** to that admission, once, to prior deadline
1792385089.2587068. Preparation/storage waits fall inside this stopped interval;
repeated export runtime counts normally and gets no additional credit.
Evaluation remains24h from first cache construction, unstarted. Same fixed
evaluation, fresh roots, cache reuse, global replay/reproduction barrier and
archive/readback/index/final unpacked-retention obligations remain. Inactive
models are restored only as needed with hash checks and HDD originals retained.

Declare exact source, tests, storage receipts, new deadline and conditional ETA
in PR301 before launch. Reuse only the existing bounded webhook event task with
fresh attempt-local dedupe. Preserve both failed scientific attempts and all
engineering failures. Record every coordinator return/handback. No automatic
retry on a new failure; no new event task, release, tag, default or merge.
