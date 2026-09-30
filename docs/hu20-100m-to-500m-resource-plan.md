# Conditional HU20 100M→500M resource plan

This is a prospective plan for review, **not launch authorization**. The mature CPU pilot's Linux measurements and owner-approved full budget are still required before final resource freeze. Scientific/learning settings remain the flagship uncapped HU20 recipe: same seed lineage, regrets, iteration weighting, K1, card/history abstraction, action menu and current-policy extraction. No architecture A/B or new independent seed is implied.

## Inputs and work

Continue all three retained B100M parents, never reset to zero or omit a slow lineage:

| Seed | Actual lifetime nodes | Iteration | Parent checkpoint SHA-256 |
| --- | ---: | ---: | --- |
| 2026093001 | 100,000,029 | 246,212 | b560669df702057b9df72c90495195a60741e1c6c4b42603cc7c330e2615f64a |
| 2026093002 | 100,000,280 | 245,406 | 830c7c470de2a416b2cb030af6abb76bbb49b2757ff1a8300b0f30a75de3b965 |
| 2026093003 | 100,000,083 | 248,186 | 77bb18426aecdf6d9e83a9c8eb807f7e6671ab01ffafc94e348382a677e9df1b |

Save at first complete iteration crossing **150M, 200M, 300M and 500M total lifetime nodes**. About 1.2B additional nodes across all three lineages. Verify parents, all input/source identities and uninterrupted RNG/iteration lineage before launch. Do not select a better intermediate checkpoint. A resource failure retains partial work; no automatic restart with altered counts. Each save exports the current policy and immutable lineage/work/resource sidecars. Two alternating atomic recovery slots every 10M additional nodes or 15 minutes; verify save/reload equivalence and all transport hashes before replacing an older recovery slot. Keep major checkpoints permanently.

## Capacity and wall-time gate

The measured M4 first-seed 100M→105M slope is 30,779 new entries/5,000,174 nodes, about 6,156 entries/M. A **linear illustration**, not validated forecasting, gives roughly **3.96M entries at 500M**. The existing **3M entry safety limit may be reached near 344M total**. Do not quietly lift that limit or promise that the original guard supports 500M. Before final freeze, a prospective guard adjustment, measured memory/headroom plan and explicit owner approval are required. Hitting a guard is a retained stop, not permission to evict keys or alter learning.

M4 direct training measured 20,314 nodes/sec with 2.067 GiB process peak at 1.528M entries. Holding that throughput constant gives approximately **5.47 hours/lineage**, **16.4 hours serial** for 400M additional nodes each, excluding table-growth slowdown, saves/exports, setup, audits and evaluation. Linear memory scaling to 3.96M entries suggests about 5.36 GiB, but Python/native overhead and serialization peaks may be nonlinear; this is insufficient to guarantee an 8 GB pod. The 16 GB candidate has more headroom and is not assumed fastest or cheapest.

Final paid configuration is the parity-passing candidate with lowest measured cost per complete node **subject to future capacity/headroom**. Record actual CPU/quota/SMT: two vCPUs are not proof of two physical cores. Prefer one heavy worker per pod. The six-class pilot permits concurrent independent pods; it does not validate multiple heavy workers in one allocation. Three isolated pods, one per lineage, are an option for independent parallelism only after capped spending approval; this pilot does not measure simultaneous workers or arbitrary multi-core scaling. CPU/RAM class, immutable rental cutoff and maximum dollars remain pending until measurements. If meaningful trainer state diverges, no paid training recommendation follows.

Use measured pilot rate `r` and price `p` to quote baseline training wall time `400M/r` per lineage and compute cost `3 × p × 400M/(3600 × r)`, then add observed save/export cadence, conservative slowdown/setup/recovery reserve, storage, verification/retrieval and evaluation. Report training-only dollars/M separately from full delivered cost. Do not present current 5M throughput as a 500M guarantee.

## Evaluation and storage are part of the campaign

Keep original B100M comparison models. Proposed final-minus-own-B100M primary original-cap2 LBR: **2,048 paired two-position blocks**, unchanged four chance samples/five soft seconds; two-sided 97.5% block interval lower bound above zero. Native-pressure safeguard: **4,096 paired blocks**, 97.5% lower bound above −10 BB/100. Average three lineage contrasts inside each same deal/rotation block. All intermediate milestones are exploratory, not selectable replacements for 500M. Native replay, fresh-deal separation, independent arithmetic and absolute/role/seed results are required; report limited attacks, fallback/off-menu exposure and full-stack-loss/large-bet tails without calling bounded LBR exact exploitability.

A full proposed panel at B100/150/200/300/500M is five retained cheap contracts ×4,096 blocks, original-cap2 LBR ×2,048 blocks, and six retained secondary opponents ×256 blocks, each ×three lineages ×two positions: **721,920 hands**. Counts/schedules, new root and precision/timing gate must be prospectively frozen before playing; no new trap-opponent implementation is included here. Evaluation stays on the M4 with fixed native attack semantics; #129 training parity does not validate wall-clock-bounded LBR on another CPU. Historical #116 completed 608,256 hands, but its mixed training/evaluation clock cannot be priced as a measured per-hand cost. Obtain an outcome-free timing forecast and reserve its measured hours, audit time and artifact disk before full-budget approval.

The 5M pilot produces a 72.25 MB compressed trainer checkpoint, 40.95 MB current export and 50.32 MB detailed iteration trace. Scaling the trace linearly yields roughly **4 GB/lineage** for 400M added nodes, before milestone/recovery checkpoints, exports, logs and replay archives. Checkpoint size/growth may also change. Use at least 30 GB disposable pod disk only if the final measured storage plan leaves 8 GB free throughout; otherwise quote a larger disk explicitly. The full campaign may exceed M4 free space after retrieval: reserve owner-approved archive storage rather than silently deleting #112–#129 evidence. Include billed disk/volume lifetime and transfer cost, archive hashes, retrieval verification and shutdown/absence checks in the full cap. No rental or 500M launch is authorized by this document.
