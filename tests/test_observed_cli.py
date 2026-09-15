import json

from scripts.check_game import main


def test_headless_observation_cli(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["check_game", "--players", "4", "--hands", "3"])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["completed_hands"] == 3
    assert report["players"] == 4
    assert sum(report["net_chips"]) == 0
