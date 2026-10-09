# Proposed v0.5.0 notes — unpublished

Usable heads-up 100 BB local play with the exact #207 first-seed O1B average and
explicit bounded public-history translation enabled (512 states, 128 events).
Human restricted/free sizing, self-play spectator decisions, legal information,
fixed resets/accounting and durable independently replayable journals use the
existing local runtime. Select this candidate explicitly; v0.4.2 remains the
released default and v0.4.0/v0.4.1 behavior stays available.

This milestone uses internal checks and scripted evaluations. No suitable free
public HU100 reference model/API has been verified, so no established external
benchmark or externally established strength is claimed. v0.5.5's HU200/Slumbot
work, including independent #218, does not gate this HU100 preparation.

#215's three-seed evidence repeats loose growth and translation gains. Its
**overall recipe remains unqualified** because one adjusted tight comparison
is inconclusive. Translated pot-pressure profitability is unproven. We retain
these decisions, all uncertainty and prior failures; we do not select the best
seed, extend evaluation, train, re-extract or conduct a live benchmark.

The package includes model card, notes, install/retrieval instructions, explicit
candidate manifest/checksums and a Python-standard-library standalone verifier.
Research bytes remain outside Git. Publication, tagging, public assets, Latest
and a default change require a separate owner-approved exact-source workflow.
