import json

from scripts.play_hu20 import play, replay_history
from src.blueprint.abstraction import HU20_SCHEMA, choices
from src.blueprint.solver import HU20_GAME
from src.game.types import ActionKind


def test_terminal_session_plays_replayable_complete_hands(tmp_path, monkeypatch):
    class CheckCallSource:
        def __init__(self, index, manifest, arm):
            assert arm == "A"
            self.coverage = {"trained": 1}

        def distribution(self, view):
            menu = choices(view, free_fold=False)
            selected = next(i for i, item in enumerate(menu)
                            if item.action.kind in (ActionKind.CHECK, ActionKind.CALL))
            return menu, tuple(float(i == selected) for i in range(len(menu))), True

        def close(self):
            pass

    monkeypatch.setattr("scripts.play_hu20.WindowedDistribution", CheckCallSource)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"game": HU20_GAME, "abstraction": HU20_SCHEMA,
                                    "artifact_sha256": "a"*64}))
    visible = []

    def choose(prompt):
        menu = [line for line in visible[-8:] if line.startswith("  ")]
        return next(line.split(".")[0].strip() for line in menu
                    if "check" in line or "call" in line)

    history = tmp_path / "hands.jsonl"
    result = play(tmp_path / "index.sqlite", manifest, arm="A", session_seed=73,
                  history=history, max_hands=2, input_fn=choose, output=visible.append)
    assert result["hands"] == 2
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert [row["button"] for row in rows] == [0, 1]
    assert replay_history(history) == 2
    assert any(line.startswith("Board: ") and len(line.split()) == 6 for line in visible)
    assert any("session:" in line for line in visible)
