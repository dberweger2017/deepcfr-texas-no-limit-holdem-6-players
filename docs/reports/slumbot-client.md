# Slumbot client: complete hands against the live API

We can now play complete hands against [Slumbot](https://slumbot.com)'s public API and verify every one of them. This is the connection v0.5.5's benchmark needs. It makes no strength claim: the live hands below were played by a probe bot and a check/call bot, only to test the protocol.

## What was added

- **Transport** (`src/arena/external/http.py`): HTTPS JSON with one request at a time, a 10 s timeout, a minimum interval between requests and a request cap. Nothing is retried. A timeout raises `AmbiguousRequest`, because the service may already have acted. Error messages name only the endpoint and status, never a request or response body, since bodies carry tokens or credentials.
- **Settlement** (`settle`, `terminal_record`, `verify_terminal` in `slumbot.py`): Slumbot reveals its hole cards at the end of every hand, folds included. So each hand's result is recomputed from the public actions and both hands: folds, uncalled raises, showdowns, ties and all-in runouts. It must equal the reported `winnings`, or the hand is rejected.
- **Hand runner** (`src/arena/external/session.py`): `new_hand`, one `act` per client decision, then the terminal response. The final action string must continue exactly what the client saw and sent, and the seat and hole cards must not change.
- **Journal:** the existing hash-chained journal now also holds verified terminal records. Rotating tokens never enter it.
- **Login:** `SlumbotConnection.login` starts an account session. The credentials go to the transport only.
- **Admission:** scripted players, which read the legal actions at any depth, are admitted. Trained policies still must be HU200.
- **CLI:** `scripts/play_slumbot.py`, with check/call by default or a pinned HU200 average, a fresh journal and `summary.json` per run.

## Live checks (October 13, 2026, anonymous)

| Run | Hands | Requests | Result |
|---|---:|---:|---|
| Probe bot: mixed check/call, folds, min-raises and all-ins | 12 | 63 | Every decision response parsed into a legal observation; every independent settlement equals Slumbot's `winnings`. Endings covered: client folds, Slumbot folds, showdowns and an all-in runout. Kept as the test fixture `tests/fixtures/slumbot/live-hands-20261013.json`, without tokens. |
| `scripts.play_slumbot`, check/call | 20 | 116 | 96 decisions; every settlement verified; the journal verifies; no token in it. One hand ended on the first response: Slumbot folded its small blind before we acted. |

**Measured costs:**
- Median latency is 1.12 s per request (90th percentile 1.22 s, maximum 2.77 s), with about 5.8 requests per hand.
- With 0.5 s spacing between requests, a hand took 9.4 s: **about 2.6 hours per 1,000 hands**. The spacing now defaults to none; only one request is ever in flight. Three check/call hands then took 4.4 s each, so expect roughly 1.5 hours per 1,000 hands.
- Requests are mostly waiting on the network, so the M1 or the M4 can run other work alongside.

**Slumbot's baseline cuts the noise considerably.** Every terminal response carries `baseline_winnings`, Slumbot's own variance-reduction baseline for the hand. On the 20 check/call hands, the 95% half-width of winnings minus baseline was **252 BB/100, against 656 for raw winnings**: about 2.6× narrower, worth roughly 7× as many hands. A benchmark should report both, with the adjusted figure as the precise one. Twenty hands of a check/call bot show nothing about strength; the ratio is the point.

## Use

```
python -m scripts.play_slumbot --hands 20 --out DIR
python -m scripts.play_slumbot --hands 1000 --model AVERAGE.gz --model-sha256 SHA --out DIR
python -m scripts.play_slumbot --hands 1000 --model AVERAGE.gz --model-sha256 SHA --out DIR --login
```

With `--login`, the username and password come from the environment variables `SLUMBOT_USERNAME` and `SLUMBOT_PASSWORD`, else from the repository's git-ignored `.env` (copy `.env.example`), else from prompts, the password hidden. Neither is stored or printed. Logged-in hands count toward that account on Slumbot's site; the hands above were anonymous.

## Limits

- **No duplicate pairing:** the service deals fresh cards each hand. The baseline is the variance reduction available.
- **A request without a complete answer abandons only its hand.** Nothing is resent; the run starts a new hand. Slumbot's running session total still counts the abandoned hand, so the summary reports Slumbot's net beside our verified net, and their difference is what abandoned hands cost. More than `--max-abandoned` such hands (default 5), an error status, a protocol mismatch or the request cap stops the run. The timeout is 30 s: on Wi-Fi, an account run lost a request after 7 hands with a 10 s timeout.
- **Service identity:** the service publishes no version, so results are tied to the dates they were played.
