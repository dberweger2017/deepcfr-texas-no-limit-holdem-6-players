# v0.5.5 confirmation seeds: admission blocked

Neither seed **2026101421** nor **2026101422** has started. The M4 fails the
owner's **45 GiB free-disk admission requirement**. No detached build was
launched, so the fixed 12-hour clock has **not started**. This is a preflight
block, not a time-stopped training result or a scientific retry.

## Admission observation

At **2026-10-14 15:30:37 UTC /17:30:37 CEST**, `ssh m4` identified
`dberwegerch-m4.local`, Mac16,1, **17,179,869,184 bytes /16 GiB RAM**.
The attached workstation is the M1; no training was run there.

- Both `~/Local/.hu20-m4-research.lock` and
  `~/Local/.hu100-m4-research.lock` admitted nonblocking exclusive `flock`
  simultaneously, held only during observation and released on exit.
- Process enumeration found no campaign matches. The exact process command,
  search terms and transport exclusions are in the receipt.
- `shutil.disk_usage(~/Local)` reported **35.474 GiB free**, **9.526 GiB short**
  of the required 45 GiB; a separate `df -k` observation is retained.
- Live GitHub checks found #272 **OPEN** and #274/#275 **MERGED**. Compute
  availability does not authorize changes to #272's research root.

[Machine-readable admission receipt](hu200-v055-confirmation-artifacts/admission.json).
No jobs were signaled and no research files were copied, archived, moved,
deleted or evicted. No M4 reservation remains held.

## Scope and outstanding work

The intended training source was current main
`db344c359523d72158f44330c1fc35dddf62322e`, including #275's completed-hand
client. Preparation used a separate unsynced M1 worktree on
`feature/hu200-v055-confirmation`; an M4 worktree and ignored results root
have not been created.

The owner request and [acceptance protocol](../hu200-v055-acceptance-protocol.md)
remain the specification: seed 2026101421 through final current/average exports
and every-row audits, then seed 2026101422, with the unchanged #250 cap-growth
and #265 frozen-20B lineage. Actual training counters, checkpoint/key hashes,
per-save statistics, guards, timings and audits do not exist for either seed.
Neither seed is labelled 20B or complete.

Training requires a fresh admission check with at least 45 GiB free before the
first detached build. Its 12-hour deadline, 15-minute terminal-save reserve,
two-hour export/audit reserve and seed-2 minimum 5.5-hour admission remain
unchanged. No smaller cap, alternate machine, scientific retry, budget extension
or paid compute was substituted.

Server preparation depends on successful final export audits and is **not
started**: no model copies, caches or launchers were created on db-ocean.
Existing server sessions, inputs, caches and runs were untouched. No Slumbot
play, credentials, release, tag or default changes occurred.

No new research payloads or ZIP archives exist; the compact admission receipt
is retained in Git. Research-Cloud ZIPs, their full local readback/upload
acceptance and restoration index remain outstanding with training. The draft PR
records this blocker only and remains for the owner's decision.
