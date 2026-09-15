import pytest


def test_hidden_and_tabled_cards_render_without_private_state(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from scripts.poker_gui import PlayerWidget
    from src.game.legacy import Card

    app = QApplication.instance() or QApplication([])
    widget = PlayerWidget(0, is_human=True)
    widget.update_hand((Card(12, 3), Card(0, 0)))
    assert widget.card1.text() == "A\n♠"
    widget.update_hand((), show_all=True)
    assert widget.card1.card is None and widget.card2.card is None
    widget.close()
    app.processEvents()
