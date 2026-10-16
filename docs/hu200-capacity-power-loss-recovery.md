# PR301 accidental power-off recovery — owner approved October 16

Status: **owner approved: “aproved, you can continue.”** Recovery deployment
validation is in progress; no scientific restart has yet been performed. This
authorizes only the common5B rollback and one-hour extension below, preserving
the original scientific question, endpoints and evaluation.

## Damage assessment

The last complete scheduler sample is October16 18:19:00.665UTC, CPU54°C/VRM67°C,
with no stop requested. The final trace row is truncated and preserved. Current
boot began about18:21:15UTC; the old scheduler and event observer are gone.

- All three5B recoveries and averages remain, with all bytes/SHA256 matching
  their pre-shutdown full-audit receipts.
- Both completed10B checkpoints, seeds1421/1422, match their native saved hashes.
  Their every-row audits were interrupted, so they are not admitted recoveries.
- Seed1302 reached10,000,000,240 nodes / iteration7,864,201 in RAM, but only
  6,707,965,952 bytes of its temporary checkpoint were written. Header-only
  inspection is evidence of progress, not a valid recovery. About99m52s elapsed
  from that job's first observed start to the last trace sample.
- All three restored twin averages match the owner's exact hashes.
- The guarded post-boot readback checks11 files /76,175,040,480 bytes in168.5s,
  exit0, CPU/VRM peaks55/53°C, peak33.44MB RSS, no swap. This verifies retained
  bytes, not hardware health or completion of the interrupted10B row audits.
- Keep every interrupted file, scratch tree, log and partial trace. No deletion,
  overwriting, model selection, final hands or outcomes occurs during assessment.

[Post-boot byte readback](reports/hu200-capacity-cygnus-artifacts/power-off-byte-readback.json),
[interruption receipt](reports/hu200-capacity-cygnus-artifacts/power-off-assessment.json).

## Approved execution

1. Preserve the old attempt in place. Recheck the source/binary/K50 table hashes,
   parent hashes, boot identity, sensors, available RAM/swap, NVMe free space and
   outstanding processes. Retain the original private callback endpoint without
   exposing it in Git or archives.
2. Start all three lineages from their **own audited5B training recoveries** under
   the original owned root:
   `results/SEED/5000000000/checkpoint.gz`. Do not resume from an average, a
   partial checkpoint or either unaudited10B save. New outputs and fresh stop
   files go under `results-recovery-20261016/SEED/TARGET/`; old paths stay intact.
3. Repeat only5B→10B once per seed, then continue10B→20B. Use the identical
   qualified trainer/binary, seeds, tables, menu/R/averaging/freeze/cap settings.
   This repeats approximately15B main-stage nodes in total, in addition to the
   original60B physical main-stage-node plan; final scientific endpoints remain
   20B per seed. Approval is a single bounded exception to the no-auto-retry rule,
   not permission for later retries or other experiments.
4. Audit every rebuilt10B save fully before that seed continues. Rebuilt1421
   must match `63abc0a8098b7ac9b8c8ce2a6b121640fd681d38a9cf3a48437a8502dbed30b1`;
   rebuilt1422 must match
   `b6dd7cb280736a5c93af1590f66f68e5655abc1b5aa83de78c14aa0fa6142fa0`.
   These old full compressed hashes are byte-identity references. For1302,
   require the known10B node/iteration header identity plus all existing
   post-write and every-row checks. Any unexplained mismatch stops the campaign.
5. Use one recovery supervisor with explicit adopted5B gates and a new immutable
   source pin; never restart the old constructor or reset its clock/gates.
   Changes may affect orchestration only. Validate successful-stage adoption,
   original-parent binding, retry refusal, clock arithmetic, disjoint CPU/RSS/
   disk reservations and global stops before launch. No trainer code changes.
   Restart only the owned metadata observer, deduplicating prior receipts and
   recognizing the new recovery attempt; retain all earlier delivery evidence.
6. Keep all original guards: three two-CPU trainer reservations, at most eight
   CPUs overall; at most three one-CPU audits where resource admission fits;
   CPU85°C immediate observed stop, Thermistor15>=95°C for30s clean stop,
   admission holds at CPU>=68°C/VRM>=83°C, RSS/swap guards,50GB free floor,
   fan/resource logging. Preserve the original1MiB swap baseline and disclose
   pre- and post-reboot swap separately. No resource limit is raised.
7. Retain the already verified5B averages as the descriptive evaluation inputs.
   Keep their model hashes unchanged. After20B fully audits, run the original
   fixed, cost-admitted evaluation; fresh roots, translation, panel/direct counts,
   cache-once plan and global every-hand replay/reproduction barrier stay fixed.
   No outcome is read before that global barrier.
8. Archive the old interrupted attempt and successful recovery together with
   distinct provenance. Complete Drive acceptance/readback/index/cleanup receipts,
   plus the owner-authorized unpacked final copy on the2TB drive. The HDD is
   final retention only, never training/evaluation/cache/scratch. PR292 and all
   unrelated work remain untouched. Delete the existing event task only at full
   closeout; no future campaign or new recurring schedule.

## Approved time amendment

Keep the original clock file immutable and add a separate approved interruption
ledger. Apply **exactly+3,600s once** to the main gates/training/audit deadline:

| Limit | Original | Approved |
| --- | --- | --- |
| Main total ceiling |72h |73h |
| Main deadline UTC |October19 02:29:18 |October19 03:29:18 |
| Main deadline Madrid |October19 04:29:18 |October19 05:29:18 |
| Precise deadline epoch |1792376957.996109 |1792380557.996109 |
| Evaluation from first cache build |24h |24h unchanged |

The one-hour credit is the owner's requested rounded allowance, not a claim of
exactly one hour electrically powered off. It does not restart a72h clock,
exclude unlimited approval wait, add another60 minutes on every resume, or
extend evaluation. All later time continues to count. If the amended deadline
cannot fit the fixed remaining work, stop and report rather than shrink the
experiment or extend again.

## Disk admission and ETA

Read-only inventory at19:19UTC finds94.330GB of owned nonsymlink, inode-deduplicated
logical files and roughly221GB free NVMe. The interrupted10B trees occupy
30,676,227,567B and must remain. Original259GB training plan,20GB growth reserve
and50GB floor remain unchanged. Three20GB native-output reservations currently
fit:94.330+60+20+50=224.330GB. Do not assume this permits three later audits.

Conservative final-retention arithmetic, allowing18GB per final recovery,
11.5GB per final average and retaining interrupted evidence, gives about156.33GB
owned. Adding one23GB audit reservation,20GB growth and50GB floor gives249.33GB,
within259GB with about9.67GB margin. Two such audit reservations do not fit that
conservative state. Use actual sizes and the existing reservation formula at each
launch; allow one-to-three audits, serialize when needed, and retire only
unchanged approved redundant recoveries/current exports after verified successors.
No failed/partial files or other PR space is reclaimed to make this plan fit.

A common5B rollback adds roughly two hours of parallel native recomputation
before the planned10B audits; it does not require retraining from scratch.
Six remaining audits project to31.76 worker-hours. Depending on actual disk
admission, their wall time can range from roughly11h to32h, plus native/load/save
work and scheduling gaps. Keep9–12h allowance for the remaining native stages,
including the repeated5B→10B segments. An upper planning sum of32+12=44h from
approval remains below the proposed deadline if approval is prompt; it is not a
runtime guarantee and must be freshly admitted before launch.

Provisional training/audit completion becomes Saturday17 evening–Sunday18 evening
Madrid. Full report/archive target remains Monday19 evening, Tuesday20 upload
contingency, conditional on prompt approval, actual evaluation cost and upload/
retention-copy throughput. Requote at recovery launch and after rebuilt10B.
No scientific work was launched before owner approval. Deployment and launch
receipts are recorded separately in PR301.
