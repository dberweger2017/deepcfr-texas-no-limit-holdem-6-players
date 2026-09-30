# v0.4 — Rebuilt Poker AI Research Preview

## What you can do

Play the saved fixed-first-seed B100M heads-up 20 BB policy at a local web table, inspect public hand histories and replay private records through the native engine. Restricted research mode exposes the trained concrete actions. Experimental free sizing passes exact native-legal human wagers to the engine while the bot keeps its existing abstract policy and fallback.

## Quick start

Follow the [source install and verified model download](../../../readme.md#quick-start), then the [play guide](../../play-web.md). The service binds to loopback and requires an access token. No public hosted endpoint or PyPI release is provided.

## What changed

The repository now has a playable research entry point. The featured model is a tabular external-sampling CFR inference export after 100 million traversal nodes. The project began with neural Deep CFR; those experiments and later six-player blueprint work remain in the [research index](../../research-history.md). Merged [PR #120](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/120) adds [human benchmark sessions](../../play-web-benchmark.md) with exact planned hand counts, explicit incomplete/aborted outcomes, restart and sanitized aggregate export. These records do not establish human strength or implement AIVAT.

## Evidence

The [#116 recovery](../../reports/hu20-scaling-m4-recovery.md) and [completed diagnostics](../../reports/hu20-scaling-diagnostics.md) retain training, paired-opponent results, limitations, manifests and hashes. The release-readiness PR will record fresh-install, test, browser and replay checks against its exact candidate source SHA. The playable artifact is a **single first seed**; the principal numerical comparisons aggregate three lineages. The real-model smoke is integration evidence, not a human-strength test.

## Limitations

Only two-player, 20 BB, no-rake/no-ante play is supported by this artifact. Each hand resets stacks. Free sizing can leave the trained tree. B100M continues to lose against a bounded local response, secondary panels are mixed, and independent river coverage is thin. Six-player and 100 BB strength remain research goals. v0.5 and v1.0 criteria are unchanged. The upstream-derived engine's license remains unverified; I license my changes in the fork under MIT while seeking the original authors' terms. No MIT grant over upstream code is claimed. No training-resume checkpoints, engine binaries, private sessions or papers are in the small bundle.

## Migration from legacy releases

Older releases retain historical labels and artifacts, including inconsistent 0.x/2.x/3.x numbering. They do not establish the rebuilt roadmap's professional v1.0 standard. Use the canonical **`v0.4.0`** tag and its verified B100M asset for this preview. Do not rename an old wheel as v0.4. The proposed legacy display cleanups are listed separately in the cleanup plan.
