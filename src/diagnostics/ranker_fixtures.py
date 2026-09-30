"""Frozen, explicit rank boundaries for the independent engineering check."""

from random import Random

from src.blueprint.search import DECK


FIXTURES = (
    ("Ac Kd 9h 7s 5c 3d 2h", (0, 14, 13, 9, 7, 5)),
    ("Ac Ad Kh Qs 9c 3d 2h", (1, 14, 13, 12, 9)),
    ("Ac Ad Kh Ks Qc 3d 2h", (2, 14, 13, 12)),
    ("Ac Ad Kh Ks Qc Qd 2h", (2, 14, 13, 12)),
    ("Ac Ad Ah Ks Qc 3d 2h", (3, 14, 13, 12)),
    ("Ac 2d 3h 4s 5c Kd Qh", (4, 5)),
    ("Ac Kd Qh Js Tc 3d 2h", (4, 14)),
    ("2c 3d 4h 5s 6c 7d Kh", (4, 7)),
    ("Ac Jc 9c 7c 4c 3d 2h", (5, 14, 11, 9, 7, 4)),
    ("Ac Kc Jc 9c 7c 4c 2h", (5, 14, 13, 11, 9, 7)),
    ("Ac Ad Ah Ks Kc 3d 2h", (6, 14, 13)),
    ("Ac Ad Ah Ks Kc Kd 2h", (6, 14, 13)),
    ("Kc Kd Kh As Ac Qd Qh", (6, 13, 14)),
    ("Ac Ad Ah As Kc 3d 2h", (7, 14, 13)),
    ("2c 2d 2h 2s Ac Kd Qh", (7, 2, 14)),
    ("Ac 2c 3c 4c 5c Kd Qh", (8, 5)),
    ("Ac Kc Qc Jc Tc 3d 2h", (8, 14)),
    ("2c 3c 4c 5c 6c 7c Kh", (8, 7)),
    ("Ac 2c 3c 4c 6c 5d Kh", (5, 14, 6, 4, 3, 2)),
    ("Ac Ad Kh Ks Qc Qd Ah", (6, 14, 13)),
)


def distinct_hands(count=100000):
    random = Random(202610040101)
    order = {card: index for index, card in enumerate(DECK)}
    seen = set()
    while len(seen) < count:
        cards = tuple(sorted(random.sample(DECK, 7), key=order.__getitem__))
        if cards not in seen:
            seen.add(cards)
            yield cards
