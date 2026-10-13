"""Native HU20 trainer parity; runs when native/hu20-trainer has been built in release mode."""

import subprocess
import sys
from pathlib import Path

import pytest

BINARY = Path("native/hu20-trainer/target/release/hu20-trainer")
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="build native/hu20-trainer first")


def run(*args):
    return subprocess.run([sys.executable, *args], check=True, capture_output=True, text=True)


def test_rules_menus_and_keys_match_the_python_engine(tmp_path):
    fixtures = tmp_path / "hands.jsonl"
    run("-m", "scripts.native_parity_fixtures", "--hands", "500", "--passive", "0.6", "--out", str(fixtures))
    out = subprocess.run([str(BINARY), "parity", str(fixtures)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "mismatched_hands 0" in out.stdout


@pytest.mark.parametrize("stack_bb", [20, 100])
def test_equity_bucket_keys_match_the_python_engine(tmp_path, stack_bb):
    """The equity schema changes only the postflop card part, identically in both engines."""
    from tests.native_equity_tables import fixture_situations, write_tables
    seed, hands = 202610090100, 300
    tables = write_tables(tmp_path, fixture_situations(range(seed, seed + hands)))
    fixtures = tmp_path / "hands.jsonl"
    common = ["--hands", str(hands), "--seed", str(seed), "--passive", "0.6", "--stack-bb", str(stack_bb)]
    run("-m", "scripts.native_parity_fixtures", *common, "--card-buckets", str(tables), "--card-tables-unpinned",
        "--out", str(fixtures))
    out = subprocess.run([str(BINARY), "parity", str(fixtures), "--card-buckets", str(tables), "--card-tables-unpinned"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert f"hands {hands} " in out.stdout and "mismatched_hands 0" in out.stdout
    # Postflop keys differ from v1's; preflop keys are v1's apart from the schema name.
    v1 = tmp_path / "v1.jsonl"
    run("-m", "scripts.native_parity_fixtures", *common, "--out", str(v1))
    import json
    pairs = [(a, b) for x, y in zip(fixtures.read_text().splitlines(), v1.read_text().splitlines())
             for a, b in zip(json.loads(x)["decisions"], json.loads(y)["decisions"])]
    assert sum(a["street"] != "preflop" for a, _ in pairs) > 500
    assert all(a["key"] != b["key"] for a, b in pairs)
    assert all(a["menu"] == b["menu"] for a, b in pairs)
    # Production loads refuse tables other than #163's.
    refused = subprocess.run([str(BINARY), "parity", str(fixtures), "--card-buckets", str(tables)], capture_output=True, text=True)
    assert refused.returncode != 0 and "#163" in refused.stderr


def test_traversals_reproduce_python_deltas_bit_for_bit(tmp_path):
    fixture = tmp_path / "traversal.json"
    run("-m", "scripts.native_traversal_fixtures", "--train-iterations", "100", "--cases", "40", "--out", str(fixture))
    out = subprocess.run([str(BINARY), "traversal-parity", str(fixture)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "mismatched 0" in out.stdout


@pytest.mark.parametrize("roots", [1, 4])
def test_a_native_run_equals_the_python_run_with_the_same_seed(tmp_path, roots):
    from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
    from src.blueprint.artifact import save_training
    from src.blueprint.solver import BlueprintTrainer, HU20_UNCAPPED_GAME, PilotConfig
    from src.game.hand import Table
    config = PilotConfig(seed=2026100517, raise_cap=None, abstraction=HU20_UNCAPPED_SCHEMA, game=HU20_UNCAPPED_GAME,
                         roots_per_seat=roots, postflop_replicates=1, max_nodes=10**9, max_entries=10**9,
                         max_seconds=900)
    trainer = BlueprintTrainer(Table(("player-0", "player-1"), (2000, 2000)), config)
    for _ in range(120 // roots):
        trainer.step()
    save_training(trainer, tmp_path / "python.json.gz")
    subprocess.run([str(BINARY), "train", "--nodes", str(10**12), "--iterations", str(trainer.iteration),
                    "--seed", str(config.seed), "--roots-per-seat", str(roots), "--out", str(tmp_path / "native.json.gz")],
                   check=True, capture_output=True)
    out = run("-m", "scripts.compare_native_lineage", str(tmp_path / "python.json.gz"), str(tmp_path / "native.json.gz"))
    assert '"identical": true' in out.stdout


@pytest.mark.parametrize("bb", [20, 100, 200])
def test_opponent_sampled_average_matches_a_python_reference(tmp_path, bb):
    """Hooks Python's trainer to add t * policy at each sampled opponent node, in native's order."""
    import gzip
    import json
    from src.blueprint import solver
    from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, HU100_SCHEMA, HU200_SCHEMA
    from src.blueprint.solver import BlueprintTrainer, HU20_UNCAPPED_GAME, HU100_GAME, HU200_GAME, PilotConfig
    from src.game.hand import Table
    config = PilotConfig(seed=2026100523, raise_cap=None, abstraction={20: HU20_UNCAPPED_SCHEMA, 100: HU100_SCHEMA, 200: HU200_SCHEMA}[bb],
                         game={20: HU20_UNCAPPED_GAME, 100: HU100_GAME, 200: HU200_GAME}[bb],
                         roots_per_seat=1, postflop_replicates=1, max_nodes=10**9, max_entries=10**9, max_seconds=900)
    trainer = BlueprintTrainer(Table(("player-0", "player-1"), (bb*100, bb*100)), config)
    roots, last, average = [], [None], {}
    original_distribution, original_random, original_start = solver._distribution, solver.Random, solver.Hand.start

    def distribution(nodes, key, menu):
        policy, trained = original_distribution(nodes, key, menu)
        last[0] = (key, policy)
        return policy, trained

    class Recording(original_random):
        def choices(self, population, weights=None, *, cum_weights=None, k=1):
            key, policy = last[0]
            delta = roots[-1].setdefault(key, [0.0] * len(policy))
            for i, p in enumerate(policy):
                delta[i] += (trainer.iteration + 1) * p
            return super().choices(population, weights=weights, cum_weights=cum_weights, k=k)

    def start(cls, table, *, hand_id, seed):
        roots.append({})
        return original_start(table, hand_id=hand_id, seed=seed)

    try:
        solver._distribution, solver.Random, solver.Hand.start = distribution, Recording, classmethod(start)
        for _ in range(80):
            roots.clear()
            trainer.step()
            merged = {}
            for root in roots:  # task order, then one addition into the table, as native applies deltas
                for key, delta in root.items():
                    target = merged.setdefault(key, [0.0] * len(delta))
                    for i, v in enumerate(delta):
                        target[i] += v
            for key, delta in merged.items():
                target = average.setdefault(key, [0.0] * len(delta))
                for i, v in enumerate(delta):
                    target[i] += v
    finally:
        solver._distribution, solver.Random, solver.Hand.start = original_distribution, original_random, original_start
    out = tmp_path / "native.json.gz"
    subprocess.run([str(BINARY), "train", "--stack-bb", str(bb), "--nodes", str(10**12), "--iterations", "80", "--seed", str(config.seed),
                    "--average-rule", "opponent-sampled", "--out", str(out)], check=True, capture_output=True)
    with gzip.open(out, "rt") as f:
        header = json.loads(f.readline())
        rows = {row[0]: row[1:] for row in map(json.loads, f)}
    assert header["average_rule"] == "opponent-sampled" and header["iteration"] == 80
    assert set(rows) == set(trainer.nodes) | set(average)
    for key, (names, regrets, stored, visits) in rows.items():
        node = trainer.nodes.get(key)
        assert regrets == (node.regrets if node else [0.0] * len(names))  # regrets don't depend on the average
        assert visits == (node.visits if node else 0)
        assert stored == average.get(key, [0.0] * len(names))


def test_python_refuses_to_continue_an_opponent_sampled_checkpoint(tmp_path):
    from src.blueprint.artifact import load_training
    from src.diagnostics.cfr_average import extract
    from src.diagnostics.saved_hu20 import file_hash
    checkpoint = tmp_path / "native.json.gz"
    subprocess.run([str(BINARY), "train", "--nodes", "200000", "--seed", "4", "--average-rule", "opponent-sampled",
                    "--out", str(checkpoint)], check=True, capture_output=True)
    with pytest.raises(ValueError, match="traverser-reach"):
        load_training(checkpoint)
    import gzip
    import json
    header = json.loads(gzip.open(checkpoint, "rt").readline())
    spec = {"seed": 4, "iteration": header["iteration"], "checkpoint_sha256": file_hash(checkpoint)}
    extract(checkpoint, spec, tmp_path / "py-average.jsonl.gz")
    subprocess.run([str(BINARY), "export", str(checkpoint), "--average", str(tmp_path / "rs-average.jsonl.gz")], check=True)
    lines = lambda name: [json.loads(line) for line in gzip.open(tmp_path / name, "rt")]
    assert lines("py-average.jsonl.gz") == lines("rs-average.jsonl.gz")
    assert lines("rs-average.jsonl.gz")[0]["extraction"].endswith("opponent-sampled-accumulator-v1")


def test_python_loads_and_continues_a_native_checkpoint(tmp_path):
    from src.blueprint.artifact import export_policy, load_training
    checkpoint = tmp_path / "native.json.gz"
    subprocess.run([str(BINARY), "train", "--nodes", "200000", "--seed", "3", "--out", str(checkpoint)], check=True)
    trainer = load_training(checkpoint)
    assert trainer.nodes and trainer.iteration > 0
    for _ in range(20):
        trainer.step()  # raises if any native key carries a different menu
    export_policy(trainer, tmp_path / "current.json.gz")


def test_native_exports_equal_python_exports(tmp_path):
    import gzip
    import json
    from src.blueprint.artifact import export_policy, load_training
    from src.diagnostics.cfr_average import extract
    from src.diagnostics.saved_hu20 import file_hash
    checkpoint = tmp_path / "native.json.gz"
    subprocess.run([str(BINARY), "train", "--nodes", "300000", "--seed", "9", "--roots-per-seat", "4",
                    "--out", str(checkpoint)], check=True)
    trainer = load_training(checkpoint)
    export_policy(trainer, tmp_path / "py-current.json.gz")
    spec = {"seed": trainer.config.seed, "iteration": trainer.iteration, "checkpoint_sha256": file_hash(checkpoint)}
    extract(checkpoint, spec, tmp_path / "py-average.jsonl.gz")
    subprocess.run([str(BINARY), "export", str(checkpoint), "--current", str(tmp_path / "rs-current.json.gz"),
                    "--average", str(tmp_path / "rs-average.jsonl.gz")], check=True)
    load = lambda name: json.loads(gzip.decompress((tmp_path / name).read_bytes()))
    lines = lambda name: [json.loads(line) for line in gzip.open(tmp_path / name, "rt")]
    assert load("py-current.json.gz") == load("rs-current.json.gz")
    assert lines("py-average.jsonl.gz") == lines("rs-average.jsonl.gz")


def test_cfr_plus_checkpoints_export_and_load_like_production_ones(tmp_path):
    """--regret-floor 0 keeps production average weights: both exporters agree and arena loaders accept it."""
    import gzip
    import json
    from src.arena.catalog import Checkpoint
    from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT, load_training
    from src.blueprint.average import AveragePolicy
    from src.diagnostics.cfr_average import extract
    from src.diagnostics.saved_hu20 import file_hash
    checkpoint = tmp_path / "cfr-plus.json.gz"
    subprocess.run([str(BINARY), "train", "--nodes", "300000", "--seed", "4", "--regret-floor", "0",
                    "--out", str(checkpoint)], check=True, capture_output=True)
    lines = gzip.open(checkpoint, "rt").read().splitlines()
    header = json.loads(lines[0])
    assert header["training_options"] == "regret-floor-0" and "average_rule" not in header
    assert all(r >= 0 for line in lines[1:] for r in json.loads(line)[2])
    with pytest.raises(ValueError, match="native options"):
        load_training(checkpoint)
    current, average = tmp_path / "current.json.gz", tmp_path / "rs-average.jsonl.gz"
    subprocess.run([str(BINARY), "export", str(checkpoint), "--current", str(current), "--average", str(average)], check=True)
    spec = {"seed": 4, "iteration": header["iteration"], "checkpoint_sha256": file_hash(checkpoint)}
    extract(checkpoint, spec, tmp_path / "py-average.jsonl.gz")
    rows = lambda name: [json.loads(line) for line in gzip.open(tmp_path / name, "rt")]
    assert rows("py-average.jsonl.gz") == rows("rs-average.jsonl.gz")
    policy = FrozenBlueprint(Checkpoint("cfr-plus", str(current), file_hash(current), HU20_UNCAPPED_FORMAT), current)
    assert json.loads(gzip.open(current, "rt").read())["training_options"] == "regret-floor-0"
    assert policy.description["iteration"] == header["iteration"] and policy.entries
    diagnostic = AveragePolicy(average, file_hash(average))
    assert diagnostic.description["iteration"] == header["iteration"]


def test_zero_mass_current_fallback_matches_between_exporters_and_loads(tmp_path):
    """--zero-mass current: zero-mass average keys play their regret-matched policy, labeled and audited."""
    import gzip
    import json
    from src.blueprint.solver import regret_match
    from src.blueprint.average import AveragePolicy, ZERO_MASS_RULES
    from src.diagnostics.cfr_average import audit, extract
    from src.diagnostics.saved_hu20 import file_hash
    checkpoint = tmp_path / "T.json.gz"
    subprocess.run([str(BINARY), "train", "--nodes", "300000", "--seed", "4", "--out", str(checkpoint)], check=True, capture_output=True)
    current, average = tmp_path / "current.json.gz", tmp_path / "rs-average.jsonl.gz"
    subprocess.run([str(BINARY), "export", str(checkpoint), "--current", str(current), "--average", str(average),
                    "--zero-mass", "current"], check=True, capture_output=True)
    header = json.loads(gzip.open(checkpoint, "rt").readline())
    spec = {"name": "T", "seed": 4, "iteration": header["iteration"], "checkpoint_sha256": file_hash(checkpoint),
            "sha256": file_hash(current)}
    extract(checkpoint, spec, tmp_path / "py-average.jsonl.gz", zero_mass="current")
    rows = lambda name: [json.loads(line) for line in gzip.open(tmp_path / name, "rt")]
    assert rows("py-average.jsonl.gz") == rows("rs-average.jsonl.gz")
    assert rows("rs-average.jsonl.gz")[0]["zero_mass_rule"] == ZERO_MASS_RULES["current"]
    audit(checkpoint, current, average, spec, file_hash(average))
    stored = {r[0]: r for r in (json.loads(line) for line in gzip.open(checkpoint, "rt").read().splitlines()[1:])}
    fallback = [r for r in rows("rs-average.jsonl.gz")[1:] if r[3] == 0]
    assert fallback and all(r[2] == list(regret_match(tuple(stored[r[0]][2]))) for r in fallback)
    assert any(r[2] != [1 / len(r[1])] * len(r[1]) for r in fallback)
    policy = AveragePolicy(average, file_hash(average))
    assert policy.description["zero_mass_rule"] == "current" and len(policy.zero_mass) == len(fallback)


@pytest.mark.parametrize("stack_bb,equity", [(20, False), (100, False), (200, False), (100, True)])
def test_overbet_menu_keys_match_the_python_engine(tmp_path, stack_bb, equity):
    """The v2 menu adds half-pot and 1.5x and 2x pot raises, with history labels for each size."""
    import json
    seed, hands = 202610130100, 300
    common = ["--hands", str(hands), "--seed", str(seed), "--passive", "0.4", "--stack-bb", str(stack_bb)]
    cards = []
    if equity:
        from tests.native_equity_tables import fixture_situations, write_tables
        tables = write_tables(tmp_path, fixture_situations(range(seed, seed + hands)))
        cards = ["--card-buckets", str(tables), "--card-tables-unpinned"]
    v2, v1 = tmp_path / "v2.jsonl", tmp_path / "v1.jsonl"
    run("-m", "scripts.native_parity_fixtures", *common, *cards, "--menu", "v2", "--out", str(v2))
    out = subprocess.run([str(BINARY), "parity", str(v2), *cards, "--menu", "v2"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert f"hands {hands} " in out.stdout and "mismatched_hands 0" in out.stdout
    names = {name for line in v2.read_text().splitlines() for d in json.loads(line)["decisions"] for name, _ in d["menu"]}
    assert {"half", "pot", "pot-1.5", "pot-2"} <= names
    # The same seeds under v1 give other menus and keys; v1's own parity is unchanged.
    run("-m", "scripts.native_parity_fixtures", *common, *cards, "--out", str(v1))
    first = lambda path: json.loads(path.read_text().splitlines()[0])["decisions"][0]
    assert first(v1)["key"] != first(v2)["key"]
    out = subprocess.run([str(BINARY), "parity", str(v1), *cards], capture_output=True, text=True)
    assert out.returncode == 0 and "mismatched_hands 0" in out.stdout


def test_menu_v2_checkpoints_name_their_menu_resume_only_with_it_and_load_in_python(tmp_path):
    import gzip
    import json
    from src.blueprint.abstraction import HU100_MENU_V2_SCHEMA, HU100_SCHEMA
    from src.blueprint.average import AveragePolicy
    from src.diagnostics.saved_hu20 import file_hash
    from src.game.hand import Hand, Table
    train = lambda *args: subprocess.run([str(BINARY), "train", "--stack-bb", "100", "--seed", "5",
                                          "--average-rule", "opponent-sampled", *args], capture_output=True, text=True)
    full, half, resumed = tmp_path / "full.json.gz", tmp_path / "half.json.gz", tmp_path / "resumed.json.gz"
    assert train("--menu", "v2", "--nodes", "400000", "--out", str(full)).returncode == 0
    assert train("--menu", "v2", "--nodes", "200000", "--out", str(half)).returncode == 0
    resume = ["--resume", str(half), "--resume-sha256", file_hash(half), "--nodes", "400000"]
    assert train("--menu", "v1", *resume, "--out", str(tmp_path / "v1.json.gz")).returncode != 0
    assert train("--menu", "v2", *resume, "--out", str(resumed)).returncode == 0
    assert file_hash(full) == file_hash(resumed)
    header = json.loads(gzip.open(full, "rt").readline())
    assert header["abstraction"] == header["config"]["abstraction"] == HU100_MENU_V2_SCHEMA
    assert header["identity"]["action_menu"] == "hu100-min-half-pot-overbet-conditional-jam-native-reopening-v2"
    average = tmp_path / "average.jsonl.gz"
    subprocess.run([str(BINARY), "export", str(full), "--average", str(average)], check=True, capture_output=True)
    with pytest.raises(ValueError):
        AveragePolicy(average, file_hash(average), expected_schema=HU100_SCHEMA, sorted_rows=True)
    model = AveragePolicy(average, file_hash(average), expected_schema=HU100_MENU_V2_SCHEMA, sorted_rows=True)
    from random import Random
    found = 0
    for seed in range(40):   # a short run has trained some preflop classes, not all
        deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
        Random(seed).shuffle(deck)
        hand = Hand.from_deck(Table(("a", "b"), (10000, 10000), button=0), hand_id="x", deck=tuple(deck))
        menu, probabilities, known = model.distribution(hand.observe(hand.actor))
        # The opening menu: half-pot equals the min-raise there, so it is skipped.
        assert [c.name for c in menu] == ["fold", "call", "min", "pot", "pot-1.5", "pot-2"]
        assert abs(sum(probabilities) - 1) < 1e-9
        found += known
    assert found > 0
