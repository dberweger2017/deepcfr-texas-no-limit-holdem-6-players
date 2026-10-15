# v0.5.3: frozen heads-up 100 BB K50 menu v2

Unpublished preparation; translation science and runtime smoke are pending.
No release or publication is authorized by this document.

The declared candidate is PR272's frozen 5B opponent-sampled average, seed
2026100601. The model uses K50 postflop cards and a broader wager menu: minimum,
half-pot, pot, 1.5× pot, 2× pot and conditional all-in, with duplicate targets
removed. Its table is frozen at 97,290,578 entries.

PR296 completed the frozen three-seed evaluation. With v2 translation off and
v0.5.2 translation on, direct gains were +47.04 [40.13,53.94], +40.95
[34.02,47.87] and +45.42 [38.49,52.36] BB/100. No scripted matchup had a negative
estimated absolute return; margins against half-pot and matched-pot pressure
were smaller. The original seed is the declared candidate, without seed selection.

Release preparation adds a distinct public-history translation identity for
human off-menu bets. Adoption requires the predeclared seven-opponent safeguard.
The translation result, final asset hashes and M1 smoke measurements remain
pending and must be recorded before readiness is asserted.

Scripted opponents only; no external HU100 benchmark. Intervals condition on
each fixed training seed and policy pair. These results do not establish human,
external-bot, multiplayer or exploitability performance.

HU20's default remains v0.4.2. No tag or release is created by packaging.
