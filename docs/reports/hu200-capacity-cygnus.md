# HU200 capacity experiment on cygnus-x1

Preflight in progress; **no large training or final hands have started**.
[Protocol](../hu200-capacity-cygnus-protocol.md) fixes three paired seeds,
planned301,153,954-entry capacity, thermal/resource guards and final sample sizes.
The capacity and safe storage schedule remain subject to measured admission.

October15 initial read-only inventory:134,984,126,464 bytes visible RAM,
132,203,900,928 bytes available, zero swap used,353,836,523,520 bytes free disk.
CPU36°C, Thermistor15 30.5°C, fan1 340RPM. No campaign workers observed;
exclusive compute lock is free. #292 remains open; its58,415,280,128-byte root,
archives and upload processes are protected and remain untouched.

The connected Drive download of the seed1421 average ZIP failed before action
invocation with HTTP413:3,162,672,661 bytes exceeds268,435,456-byte limit.
The established read-only synced-archive/SSH route is now transferring inputs.
All extraction/hashing/model work is on bb128. Restoration verification remains
pending; no model identity or strength result is claimed. Three guard admission
tests pass on bb128, and the bounded zero-step500M memory re-save is running.

Initial ETA: preflight/admission pending; allow up to72h training/audits and24h
evaluation after admission, plus bandwidth-dependent archive closeout. Update
the PR description after measured cost/storage admission. Owner decides merge.
