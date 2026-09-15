"""Independent sequence-form linear programs for the two small zero-sum games."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import bmat, coo_matrix, csr_matrix

from src.solver.tree import GameTree


@dataclass(frozen=True)
class Equilibrium:
    lower_value: float
    upper_value: float
    policy: np.ndarray
    maximum_residual: float


def solve(tree: GameTree) -> Equilibrium:
    sequences = [{(): 0}, {(): 0}]
    action_sequences = {}
    for info, key in enumerate(tree.information_sets):
        owner = key.player
        prefix = tree.own_sequences[info]
        for action in range(len(key.actions)):
            sequence = prefix + ((info, action),)
            sequences[owner][sequence] = len(sequences[owner])
            action_sequences[info, action] = sequences[owner][sequence]
    constraints, targets = [], []
    for player in (0, 1):
        rows, columns, data = [0], [0], [1.0]
        count = 1
        for info, key in enumerate(tree.information_sets):
            if key.player != player:
                continue
            rows.append(count)
            columns.append(sequences[player][tree.own_sequences[info]])
            data.append(-1.0)
            for action in range(len(key.actions)):
                rows.append(count)
                columns.append(action_sequences[info, action])
                data.append(1.0)
            count += 1
        constraints.append(
            coo_matrix(
                (data, (rows, columns)), shape=(count, len(sequences[player]))
            ).tocsr()
        )
        targets.append(np.eye(1, count)[0])
    rows, columns, payoffs = [], [], []

    def visit(node, sequence_ids, chance_reach):
        actor = tree.actor[node]
        if actor == -2:
            rows.append(sequence_ids[0])
            columns.append(sequence_ids[1])
            payoffs.append(chance_reach * tree.terminal_values[node])
            return
        for action, child in enumerate(tree.children[node]):
            following = list(sequence_ids)
            probability = chance_reach
            if actor == -1:
                probability *= tree.chance[node][action]
            else:
                following[actor] = action_sequences[tree.info[node], action]
            visit(child, tuple(following), probability)

    visit(0, (0, 0), 1.0)
    payoff = coo_matrix(
        (payoffs, (rows, columns)), shape=(len(sequences[0]), len(sequences[1]))
    ).tocsr()
    e, f = constraints
    a, b = targets
    # Maximize b.v with A.T x >= F.T v, E x = a; solve the dual separately.
    maximizing = linprog(
        np.r_[np.zeros(e.shape[1]), -b],
        A_ub=bmat([[-payoff.T, f.T]]),
        b_ub=np.zeros(f.shape[1]),
        A_eq=bmat([[e, csr_matrix((e.shape[0], f.shape[0]))]]),
        b_eq=a,
        bounds=[(0, None)] * e.shape[1] + [(None, None)] * f.shape[0],
        method="highs",
    )
    minimizing = linprog(
        np.r_[np.zeros(f.shape[1]), a],
        A_ub=bmat([[payoff, -e.T]]),
        b_ub=np.zeros(e.shape[1]),
        A_eq=bmat([[f, csr_matrix((f.shape[0], e.shape[0]))]]),
        b_eq=b,
        bounds=[(0, None)] * f.shape[1] + [(None, None)] * e.shape[0],
        method="highs",
    )
    if not maximizing.success or not minimizing.success:
        raise ArithmeticError(
            f"Sequence-form solve failed: {maximizing.message}; {minimizing.message}"
        )
    realization = (maximizing.x[: e.shape[1]], minimizing.x[: f.shape[1]])
    policy = tree.mask / tree.mask.sum(axis=1, keepdims=True)
    for info, key in enumerate(tree.information_sets):
        weights = np.maximum(
            0,
            [
                realization[key.player][action_sequences[info, action]]
                for action in range(len(key.actions))
            ],
        )
        if weights.sum() > 0:
            policy[info] = 0
            policy[info, : len(weights)] = weights / weights.sum()
    tree.validate_policy(policy)
    residual = max(
        np.max(np.abs(maximizing.eqlin.residual)),
        np.max(np.abs(minimizing.eqlin.residual)),
        max(0, -np.min(maximizing.ineqlin.residual)),
        max(0, -np.min(minimizing.ineqlin.residual)),
    )
    return Equilibrium(
        float(-maximizing.fun), float(minimizing.fun), policy, float(residual)
    )
