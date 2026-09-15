"""Public information features and stable fold/check-call/raise output slots."""

import numpy as np

from src.solver.games import Action, InformationSet

FEATURES = 48
ACTION_SLOT = {Action.FOLD: 0, Action.CHECK: 1, Action.CALL: 1, Action.RAISE: 2}
HISTORY_SLOT = {action: index for index, action in enumerate(Action)}


def legal_mask(info: InformationSet) -> np.ndarray:
    mask = np.zeros(3, dtype=bool)
    for action in info.actions:
        mask[ACTION_SLOT[action]] = True
    return mask


def encode(info: InformationSet) -> np.ndarray:
    if not isinstance(info, InformationSet):
        raise TypeError("The neural encoder accepts player information sets only")
    if info.game not in {"kuhn", "leduc"} or len(info.history) not in (1, 2):
        raise ValueError("Unsupported small-game information set")
    features = np.zeros(FEATURES, dtype=np.float32)
    features[0 if info.game == "kuhn" else 1] = 1
    features[2 + info.player] = 1
    features[4 + info.card] = 1
    features[7 + (3 if info.board is None else info.board)] = 1
    features[11 + len(info.history) - 1] = 1
    for street, actions in enumerate(info.history):
        if len(actions) > 4:
            raise ValueError("Reference betting history exceeds four actions per round")
        for position, action in enumerate(actions):
            features[13 + street * 16 + position * 4 + HISTORY_SLOT[action]] = 1
    features[45:] = legal_mask(info)
    return features
