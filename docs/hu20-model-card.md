# Heads-up 20BB blueprint model card

**Status:** experimental training in preparation. No checkpoint is promoted.

**Supported play:** heads-up fixed-stack Hold'em, 20 BB per hand, 0.5/1
blinds, standard deck, no ante or rake, and the saved restricted action menu.
The human-play command presents exactly that menu. Each hand resets stacks
and alternates the button. The inference artifact must pass SHA-256 and
versioned game/schema checks before play.

**Learning method:** ordinary K1 external-sampling tabular regret updates
from zero regrets, with three independent seeds. The candidate extraction
will be frozen using development data before fresh confirmation.

**Information:** only the acting player's private cards and public
observations enter policy lookup. The postflop descriptor and ordered public
history are deliberately retained from the earlier baseline. Different
underlying histories can share an abstract key. No exact full-game
exploitability or textbook convergence guarantee is claimed for this
imperfect-recall abstraction.

**Known limits:** a fixed 20 BB stack, heads-up only, restricted raise sizes,
two-raise cap, no ante/rake or tournament payouts, and coarse postflop card
features. A manual session is usability feedback, not a strength estimate.
The model must not be used as a six-player checkpoint.

**Results and artifacts:** pending the bounded M4 campaign. This card will
be updated with the final checkpoint hashes, extraction choice, legal-hand
count, paired effects and latency before the PR is reviewed.
