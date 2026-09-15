# Fresh readiness confirmation

**Status: running. No readiness outcome yet.**

The [frozen protocol](../neural-readiness.md) is executing at revision
`d41a0540e7f6caa093799d382e3900f8503cf391` (PR #54). All eight reserved
Leduc runs began from scratch at **2026-09-15 18:23:52 UTC**; Kuhn jobs will fill
released worker slots. Every declared seed and final paired comparison will be
retained. No intermediate result selects a checkpoint or changes the recipe.

## Launch evidence

- Runpod CPU pod `82elkrnv9bhe24`, named `neural-readiness`, EUR-IS-1.
- AMD EPYC 4564P, 16 assigned vCPUs, 32 GB RAM, 20 GB temporary disk;
  eight single-threaded workers. No GPU or network volume.
- Python 3.11.16, PyTorch 2.5.1+cpu, NumPy 1.26.4, SciPy 1.17.1.
- Remote checkout verified clean and at the frozen revision before launch.
- Six short host smoke jobs across both games completed on non-confirmation
  seeds 101, 103 and 107. These are unscored implementation checks.
- All eight Leduc workers were active, each using approximately one CPU, at
  the first process check. No failure was reported.

The empty container needed Python/pip setup and the existing SSH public key.
Initial provider API requests were rejected; the working GraphQL client uses
an explicit User-Agent and the automatically supplied pod-scoped key. No
account-wide key was created. The operational allocation probe was corrected
for cgroup v1 **before** any confirmation training began. There was no repeated
confirmation attempt. The temporary web terminal was disabled after SSH setup.

## Budget and cleanup

The console quoted **$0.563/hour**: $0.56 compute plus $0.003 temporary storage.
The displayed account balance before this rental was $9.49, recorded separately
from our conservative experiment-cost accounting. Prior conservative usage was
$1.43 of the $10 CPU authorization.

This campaign's ceiling remains **$2.50 and two hours of rental**, including
setup and cleanup. Conservatively count rental from 18:15:37 UTC; the hard
cleanup deadline is **20:15:37 UTC**. At the quoted rate, two hours would cost
$1.126 before any separately assessed charges. Actual duration and final cost
will be recorded after cleanup.

A detached provider-stop watchdog is armed for 20:14:37 UTC, independent of the
training process and SSH session. API authentication and current-pod identity
were checked successfully; actual shutdown will be verified in the provider
console. The [Runpod management API](https://docs.runpod.io/sdks/graphql/manage-pods)
provides the stop operation. An independent local collector watches for completion,
downloads and hash-checks the complete archive, then requests early stop. The
scheduled task follow-up handles provider verification, termination and reporting.
No credentials are included in the artifact archive.

Stopping the process alone does not stop billing. On completion or failure,
verify the downloaded evidence before disposing of the temporary disk, terminate
the pod, and confirm that no billable storage remains. At the hard deadline,
stop compute even if retrieval is incomplete and preserve the resulting limitation.

## Pending results

Publish all sixteen final seed results, all eight paired Leduc controls and
regressions, failure records, artifact/source hashes, runtime and cost. A readiness
pass requires every seed to meet the frozen absolute limits. It does not revise
the historical failed selection screen or establish professional Hold'em strength.
