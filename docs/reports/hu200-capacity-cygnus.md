# HU200 capacity experiment on cygnus-x1

Post-BIOS preflight is complete; **no large training or final hands have started**.
[Protocol](../hu200-capacity-cygnus-protocol.md) fixes three paired seeds,
301,153,954-entry capacity, guards and final sample sizes. Owner decides merge.

The owner stopped bb128 for BIOS work and resumed October16 after adding five
fans, modifying the curve and enabling XMP3200. XMP is owner-reported; memtest86+
remains incomplete as accepted. The new2TB disk is excluded. #292 remains open;
its58,415,280,128-byte root, archives, uploads and shared dependencies remain untouched.
Unrelated preservation files remain untouched. Restart NVMe free263.760GB;
post-audit free259.927GB. [Interrupted-clock receipt](hu200-capacity-cygnus-artifacts/owner-interruption-restart.json).

The no-step500M re-save preserves all69,084,815 keys and row bytes; the only
header difference is config.max_entries. No traversal was repeated after reboot.
[Full identity](hu200-capacity-cygnus-artifacts/memory-row-identity.json).
The resumed export/audit passes on **all69,084,815 rows**, independently checking
checkpoint bytes/SHA256, current regrets and normalized average accumulators.
[Audit and exact hashes](hu200-capacity-cygnus-artifacts/memory-postbios-audit.json).
All old partial exports and the truncated final pre-shutdown resource row remain.

Measured save RSS7,528,763,392 bytes; sampled family peak7,572,365,312 bytes.
After1,384,031,032 bytes of fixed card tables, this is89.6bytes/key; allocating
all100,690,818 currently absent regret slots adds11.66bytes/key, giving about
101.3bytes/key at the observed menu mixture. Admission reserves130bytes/key,
including a five-action fully aged table, index/chunk growth and save workspace.
Three planned tables plus three card-table residents,2GiB helper and6GiB
available-memory floor total130,192,069,748 bytes. Admission must recheck actual
available RAM and the conservative forecast for all three fresh500M prefixes.
No capacity or outcome result is inferred from the zero-step measurement.

The preliminary operation used4,734.229 active seconds of its original5,400:
1,285 before shutdown and3,449.229 after restart. The measured owner-stop interval
of10,429.086seconds is excluded. The72h gates/training/audit clock has not started;
24h evaluation follows. Absolute deadlines and measured ETA are posted at admission.
Archive closeout is separately bandwidth-dependent.

[Resource summary](hu200-capacity-cygnus-artifacts/memory-resource-summary.json):
pre-change CPU/VRM peaks65/60°C, fan1 335–645RPM; resumed export/audit54/58°C,
fan1 479–722RPM. These are different phases, not a controlled cooling comparison.
No recorded resource breach. Twenty-one post-BIOS guard/identity/retention/audit
tests pass, including differential checks against the original every-row auditor.

The revised disk plan reserves259GB including50GB free floor, serial sorted-stream
audits and per-seed retirement of redundant recoveries after an audited successor.
All actual writes require fresh admission. Evaluation uses two seeds concurrently,
then the third, within the declared258.4GB plan. No other PR reclamation is assumed.

Restoration is pending. Connected Drive raw and streamed-reference requests both
reject the3.16GB archive with HTTP413 before invocation (256MiB limit). Read-only
synced-archive/SSH copies timed out; partial bytes remain, and explicit transport
resumption continues with longer connection tolerance. All hashing, extraction,
training and evaluation run on bb128. This is an input-transfer failure, not a
model or training failure. No strength result or restored-input verification is claimed.
