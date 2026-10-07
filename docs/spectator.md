# Local bot-vs-bot spectator

Issue [#183](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/issues/183), scope frozen before implementation on October 7, 2026.

## Scope and checks

Heads-up, 20 BB, no rake/ante, the existing native rules and policy menus. Reset stacks each hand and alternate the button, starting with Bot A. Choose any two compatible pinned releases, including the same release twice. Preserve human play and v0.4.0 selection. No training, search, paid compute or strength qualification.

Implementation order: pin both identities and manifests; add durable one-decision sessions; record exact inference and observations; add paused playback/step/history; audit distributions, legal actions and settlements and check the real browser. One feature PR; independently review the information boundary and run required CI before merging.

## Play and inspect

Start the existing local server with both pinned models as described in the [readme](../readme.md). Choose **Bot-vs-bot spectator**, select Bot A and Bot B, and create the session. It starts paused. **Step** plays one decision (dealing a hand first if needed); **Play** continues at the selected pace and deals the next hand after settlement. **Pause** lets any already submitted decision finish and prevents another request. Reloading resumes paused. Completed and active hands retain their decision records.

The table shows cards in two explicitly labeled bot perspectives. The decision inspector shows the acting bot's immutable observation at that decision: its cards, board, public events, stacks and legal bounds. Its probability table, selected action and lookup status come from the inference call that played that action, without re-extraction or rounded values in the journal. Opponent private cards, deck, seed, future board and evaluator data never reach either policy. The observation JSON is available for exact inspection. Hidden deck and independent per-bot sampling state live only in the private SQLite journal.

Release versions, model hashes, manifest hashes and manifest download links are frozen in session and hand history. The checked-in small manifests are byte-pinned copies of the published release assets; the existing loaders verify model bytes before startup. Both seats may share a read-only model reader, but have separate observations and sampling streams.

## Audit retained sessions

```bash
python -m scripts.audit_spectator \
  --database results/play-web/spectator/private.sqlite \
  --models-dir models
```

This read-only audit reconstructs every decision and settlement directly through the game engine, without using the service's replay or presentation helpers. It compares recorded observations and exact distributions with freshly loaded pinned policies, verifies lookup status, legal selected actions, sampling continuity, seat rotation, public-event hashes and totals. It also audits the retained prefix of an unfinished hand. These counts are interface correctness evidence, not a poker-strength test.
