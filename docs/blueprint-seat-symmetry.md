# Blueprint seat lookup compatibility

The original `blueprint-abstraction-v1` key puts its actor and public action
seats in button-relative coordinates but stores folded/all-in status in
absolute seat order. `legacy-v1` retains those exact bytes and remains the
default for historical replay. `button-zero-compatible-v1` orders that status
vector from the button before hashing. It leaves the original schema string,
action menu names, card abstraction and history encoding in the payload.
Consequently button-zero keys and action probabilities are byte-identical to
the old lookup, while rotated observations query the button-zero-trained key.
No checkpoint or export is rewritten.

The compatibility mode is restricted to the hash-pinned button-zero lineage:
the 5,834,622-entry M4 source, the 12M checkpoint, and the 58,015,659-entry
checkpoint-0.4 final artifact. The loaded training table must also have six
100-BB seats, button zero, the original schema and the expected iteration.
A table header alone does not prove all historical training used one button;
the three hashes are allowed because their retained campaign records document
that fixed table. Other checkpoints are rejected until their provenance is
reviewed. The distribution interface used by direct play and search/range
consumers selects the mode explicitly. Experiment manifests record the mode
and checkpoint hash. Historical tools keep `legacy-v1` by default.

This is a lookup adapter for immutable old tables, not a new training schema.
Future training should use a separately versioned on-disk abstraction that
canonically orders every seat-dependent field at collection and inference.
It must not write new keys under the old `blueprint-abstraction-v1` label.

The independent no-free-fold wrapper acts *after* either lookup. It retains
the original action menu for hashing, zeros fold only when checking is legal,
renormalizes, and chooses check when all original mass was on fold. It does
not alter the training action abstraction. Its eligible, changed, all-fold
and removed-mass counters are evaluation diagnostics.

Native six-seat coupled-rotation tests rotate the table, button, seats,
private cards, stacks and observed action sequence together. They include an
all-in, fold, off-menu raise and all four betting streets. The tests also
compare the same hero observations under different unseen opponent deals.
