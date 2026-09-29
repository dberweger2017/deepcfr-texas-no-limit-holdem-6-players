# Nonuniform river-range development amendment

This separately frozen [amendment](../../configs/blueprint/river-range-amendment-m4.json)
ran after the [first development result](river-development-m4.md) revealed
four effectively uniform full-range priors. It is exploratory development,
not fresh confirmation. Source `73e28e2`, the same immutable 12M checkpoint,
solver, action menu and 5/15/30/60-second measurements were used. The
[raw rows, profiles, manifest, result and checksums](river-range-amendment-m4/)
were copied from the M4 and verified. All three cases completed in 237.33
seconds under the 420-second limit; peak process RSS was 7.13 GiB under
10.5 GiB, and system swap did not increase.

The named range shapes multiply each seat's blueprint marginal by a fixed
public rule, then renormalize and apply the same card-collision mask. They
are stipulated stress cases, not estimates of real opponent frequencies.

| Root | Pot / stack (BB) | Range shape | Effective holdings per seat | 5 s | 15 s | 30 s | 60 s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Dry board | 2 / 9 | Suited ×4 | 296 / 159 | 2.910% | 1.610% | 0.929% | 0.500% |
| Low draw | 12 / 4 | Pocket pair ×4 | 796 / 796 | 0.399% | 0.131% | 0.070% | 0.036% |
| Deep mixed board | 32 / 84 | J/Q/K/A holding ×3 | 880 / 880 | 2.029% | 0.830% | 0.401% | 0.178% |

Values are average-profile exploitability divided by the root pot in each
restricted declared river game. The 30-second solves completed 670, 1,271
and 863 full sweeps. Across all eight full-range development roots (including
the first run), median exploitability was 0.262% at 30 seconds and 0.156%
at 60 seconds; maxima were 0.929% and 0.500%. Only 3/8 roots were at or
below the proposed 0.1% threshold at either time. The board-playing-straight
root is nearly trivial and is visible rather than hidden in those summaries.

**Development choice for a fresh comparison:** average extraction with a
30-second per-root decision budget and a 100,000-sweep safety cap. Relative
to 15 seconds, 30 seconds materially reduces the median and worst observed
exploitability. Sixty seconds improves it further, but doubles decision
latency and still misses the proposed threshold on five of eight roots.
This is a practical pilot choice, not a claim that 30 seconds solves every
river root. The forthcoming paired comparison must test whether the chosen
strategy produces better conditional river returns than corrected rollout;
these development best responses cannot answer that question.
