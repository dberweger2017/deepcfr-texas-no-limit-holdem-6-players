# Resource-only seven-arm preflight

Two separately frozen M4 preflights used the same 8 scripted and 4 random
rotation blocks, seven policy arms, six seat rotations per block and the
hash-pinned 5.83M/12M checkpoints. The evaluator completed the hands and
checked chip conservation, but **did not save or inspect returns**. The
[5.83M artifacts](blueprint-seat-preflight-m4/5m/) and
[12M artifacts](blueprint-seat-preflight-m4/12m/) retain every attempt,
resource progress, reached-decision counters, manifests and checksums.
The copied files matched the saved checksums.

| Checkpoint | Clean source | Hands | Failures | Elapsed, including load | Peak process RSS | Swap increase |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 5.83M | `d91b04d` | 504 | 0 | 22.77 s | 4.89 GiB | No |
| 12M | `9808736` | 504 | 0 | 61.44 s | 6.47 GiB | No |

Both runs fit the 10.5-GiB guard. The fixed confirmation scale of 1,024
scripted and 256 random blocks per checkpoint is comfortably within the
10-hour M4 experiment ceiling by extrapolation from complete preflight work;
the actual run still has explicit time, RSS and free-disk guards. The
preflight cannot predict the result or guarantee memory will not grow with
distinct reached keys. It justifies freezing this workload without peeking at
returns. The 58.02M checkpoint was streamed for whole-table density; loading
it for seven-arm play would exceed the M4 process guard, so it is excluded
from the in-memory comparison.
