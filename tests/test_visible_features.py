import pytest

from src.holdem.visible_features import visible_card_features


def test_features_use_fixed_normalized_hand_values_for_wheel_and_kickers():
    wheel = visible_card_features(("Ah", "2d"), ("3c", "4s", "5h", "9d", "Kc"))
    kickers = visible_card_features(("As", "Kd"), ("2c", "7h", "8s", "9d", "Jc"))
    assert wheel[:2] == pytest.approx((4 / 8, 5 / 14))
    assert kickers[:2] == pytest.approx((0 / 8, 14 / 14))
    assert all(0 <= value <= 1 for value in wheel + kickers)


def test_board_playing_and_suit_relabeling_are_explicit_and_invariant():
    board = ("As", "Ks", "Qs", "Js", "Ts")
    features = visible_card_features(("2c", "3d"), board)
    assert features[:2] == pytest.approx(features[6:8])

    mapping = dict(zip("cdhs", "shdc", strict=True))
    relabel = lambda cards: tuple(card[0] + mapping[card[1]] for card in cards)
    assert visible_card_features(relabel(("2c", "3d")), relabel(board)) == pytest.approx(
        features
    )


def test_visible_features_reject_invalid_or_overlapping_cards():
    with pytest.raises(ValueError, match="seven distinct"):
        visible_card_features(("Ac", "Ad"), ("Ac", "7h", "8s", "9d", "Jc"))
    with pytest.raises(ValueError, match="five distinct"):
        visible_card_features(("Ac", "Ad"), ("2c", "2c", "8s", "9d", "Jc"))
    with pytest.raises(ValueError, match="standard"):
        visible_card_features(("Ac", "ZZ"), ("2c", "7h", "8s", "9d", "Jc"))


def test_features_depend_only_on_cards_visible_at_the_decision():
    first = visible_card_features(("Ac", "Kd"), ("2c", "7h", "8s", "9d", "Jc"))
    second = visible_card_features(("Ac", "Kd"), ("2c", "7h", "8s", "9d", "Jc"))
    assert first == second
