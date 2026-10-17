# v0.5.5 packaging checks

The owner approved publishing #306's exact HU200 three-seed blend after review
and merge of the packaging change. Publication follows the verified draft and
public-download procedure in [v0.5.3's publication record](../v0.5.3/PUBLICATION.md).
The owner reviews and merges this package; publication then needs no second approval
if every declared check passes.

The fixed model is 6,282,968,150 bytes, SHA256
`ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485`,
with 156,208,331 entries, PR163 K50 tables and HU200 translation at 512 states /
128 events. No retraining, re-export or selection is part of packaging.

Required evidence before review completion:

- Ordered parts no larger than 1 GiB; standalone stdlib verification of parts,
  whole model, tables, identity and source/publication binding.
- HU200 human restricted/free sizing and spectator play at 200 BB, two fresh CLI
  loads with elapsed time and peak RSS, pending-decision restart, corrupt-input
  refusal and independent replay of every retained journal.
- The HU200 loader must build its first play cache with the streaming builder,
  then verify and map it read-only on every start. Repeat the real runtime smoke
  on this path: measure cache-building first start and cached second start with
  separate process peak RSS, rerun hands/restart/refusals and journal replay,
  and refuse corrupt/mismatched caches. Publish the measured timings, RAM and
  additional cache disk space in the shipped install guide and release notes.
- Byte-identical old manifests, unchanged old trace hashes and the HU20 v0.4.2 default.
- Public release notes, model card and install guide, with the qualification,
  failed first protocol, limitations, measured requirements and internal panel.
- Three frozen 20B ingredient models, each separately hash-pinned and verified,
  proposed as a non-Latest research pre-release `hu200-frozen-20b-seeds`.
- Staged repository artifact guard, relevant tests and green final-head CI.

The evidence and completed measurements will replace this checklist before the
packaging PR is marked ready. Release assets and raw runtime journals stay outside Git.
