from copy import deepcopy
import pytest

from src.diagnostics.global_bucket_validation import global_labels, project_compact


class Table:
    def __init__(self, k, street):
        self.k, self.street = k, street

    def bucket(self, hand, board):
        return 1 if len(board) == 4 else self.k - 1


def test_reader_mapping_keeps_blockers_and_street_k():
    data = {"boards": [["2c", "3d", "4h", "5s"], ["2c", "3d", "4h", "5s", "6c"]],
            "holdings": [["Ac", "Ad"], ["6c", "Kd"]], "codes": [[0, 0], [0, 255]]}
    tables = {(k, street): Table(k, street) for k in (50, 200) for street in ("turn", "river")}
    labels, counts = global_labels(data, tables)
    assert labels["50"] == [[1, 1], [49, 65535]]
    assert labels["200"] == [[1, 1], [199, 65535]]
    assert counts["200"] == {"turn": [1], "river": [199]}
    data["codes"][1][1] = 0
    with pytest.raises(ValueError, match="card-removal"):
        global_labels(data, tables)


def test_projection_uses_public_templates_without_mutating_source(monkeypatch):
    original = {"crossfit_labels": {"0": [[5]], "1": [[6]]},
                "pool_keys": {"v1": {"t": {"0": "frozen"}}}, "tables": {"t": {"names": ["check"]}},
                "codes": [[0]], "labels": {"50": [[5]]}}
    snapshot = deepcopy(original)
    def keys(request, data, tables):
        assert request == {"nodes": ["unchanged"]}
        assert data["crossfit_labels"] == {"0": [[49]], "1": [[199]]}
        return {"v1": original["pool_keys"]["v1"], "eq50": {}, "eq50-fit0": {"t": {"49": "k50"}},
                "eq50-fit1": {"t": {"199": "k200"}}}
    monkeypatch.setattr("src.diagnostics.global_bucket_validation.add_pool_keys", keys)
    projected = project_compact({"nodes": ["unchanged"]}, original, {"50": [[49]], "200": [[199]]})
    assert original == snapshot
    assert projected["tables"] == original["tables"]
    assert set(projected["pool_keys"]) == {"v1", "eq50-fit0", "eq50-fit1"}
