"""Compile the small trees once; keep hidden states outside strategy lookup."""

from dataclasses import dataclass

import numpy as np

from src.solver.games import InformationSet, State, new_game


@dataclass(frozen=True)
class Layer:
    parents: np.ndarray
    children: np.ndarray
    actions: np.ndarray
    nodes: np.ndarray


class GameTree:
    def __init__(self, game: str):
        self.game = game
        self.states: list[State] = []
        self.information_sets: list[InformationSet] = []
        self.children: list[tuple[int, ...]] = []
        self.chance: list[tuple[float, ...]] = []
        info_ids, node_infos, depths = {}, [], []
        representatives, own_sequences = [], []
        edges: dict[int, list[tuple[int, int, int]]] = {}

        def visit(state, depth, sequences):
            node = len(self.states)
            self.states.append(state)
            self.children.append(())
            self.chance.append(())
            depths.append(depth)
            info = -1
            if state.actor >= 0:
                key = state.information_set()
                if key not in info_ids:
                    info_ids[key] = len(self.information_sets)
                    self.information_sets.append(key)
                    representatives.append(node)
                    own_sequences.append(sequences[state.actor])
                info = info_ids[key]
                if own_sequences[info] != sequences[state.actor]:
                    raise ValueError("Information set violates perfect recall")
                if depths[representatives[info]] != depth:
                    raise ValueError(
                        "Reference information sets must share a tree depth"
                    )
            node_infos.append(info)
            if state.actor == -2:
                return node
            if state.actor == -1:
                outcomes = state.chance_outcomes()
                successors = [state.deal(card) for card, _ in outcomes]
                self.chance[node] = tuple(p for _, p in outcomes)
            else:
                successors = [state.play(action) for action in state.actions()]
            children = []
            for action, successor in enumerate(successors):
                following = list(sequences)
                if state.actor >= 0:
                    following[state.actor] += ((info, action),)
                child = visit(successor, depth + 1, tuple(following))
                children.append(child)
                edges.setdefault(depth, []).append((node, child, action))
            self.children[node] = tuple(children)
            return node

        visit(new_game(game), 0, ((), ()))
        self.actor = np.array([s.actor for s in self.states])
        self.info = np.array(node_infos)
        self.representatives = np.array(representatives)
        self.owners = self.actor[self.representatives]
        self.own_sequences = tuple(own_sequences)
        self.mask = np.array(
            [[a < len(i.actions) for a in range(3)] for i in self.information_sets]
        )
        self.terminal_values = np.array(
            [s.returns()[0] if s.terminal else 0.0 for s in self.states]
        )
        self.layers = []
        self.chance_edges = np.ones(len(self.states))
        for depth in sorted(edges):
            parents, children, actions = np.array(edges[depth]).T
            self.layers.append(Layer(parents, children, actions, np.unique(parents)))
            for parent, child, action in edges[depth]:
                if self.actor[parent] == -1:
                    self.chance_edges[child] = self.chance[parent][action]

    def validate_policy(self, policy: np.ndarray) -> None:
        if (
            policy.shape != self.mask.shape
            or not np.isfinite(policy).all()
            or (policy < 0).any()
            or (policy[~self.mask] != 0).any()
            or not np.allclose(policy.sum(axis=1), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError(
                "Expected finite normalized probabilities on legal actions"
            )

    def edge_probabilities(self, policy: np.ndarray) -> np.ndarray:
        probabilities = self.chance_edges.copy()
        for layer in self.layers:
            decision = self.actor[layer.parents] >= 0
            parents = layer.parents[decision]
            probabilities[layer.children[decision]] = policy[
                self.info[parents], layer.actions[decision]
            ]
        return probabilities

    def reaches(self, probabilities: np.ndarray) -> np.ndarray:
        # Columns are each player's own reach and chance reach, kept separate.
        reach = np.ones((len(self.states), 3))
        for layer in self.layers:
            reach[layer.children] = reach[layer.parents]
            owners = self.actor[layer.parents]
            columns = np.where(owners < 0, 2, owners)
            reach[layer.children, columns] *= probabilities[layer.children]
        return reach

    def values(self, probabilities: np.ndarray) -> np.ndarray:
        values = self.terminal_values.copy()
        for layer in reversed(self.layers):
            totals = np.bincount(
                layer.parents,
                weights=probabilities[layer.children] * values[layer.children],
                minlength=len(values),
            )
            values[layer.nodes] = totals[layer.nodes]
        return values
