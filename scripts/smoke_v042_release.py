"""Small deterministic runtime/replay smoke; no arena or playing-strength estimate."""

import argparse
import gc
import json
from pathlib import Path
from random import Random
from unittest.mock import patch

from scripts.audit_v041_web import audit_states as audit_human
from src.blueprint.average import AveragePolicy
from src.play_api.spectator_audit import audit_states as audit_spectator
from src.play_api.versions import load_tables
from src.policies.v042 import MODEL_SHA256, ASSET_NAME


def smoke(models: Path, out: Path, source: str) -> dict:
    if out.exists():
        raise ValueError("Preserve existing smoke evidence; choose a fresh destination")
    out.mkdir(parents=True)
    runtime = load_tables(models, out / "data", source)
    human_states = []
    rng = Random(202610080042)
    try:
        catalog = runtime.model_catalog()
        assert catalog["default"] == "v0.4.2"
        assert {m["version"] for m in catalog["models"]} == {"v0.4.0", "v0.4.1", "v0.4.2"}
        assert runtime.model_info()["sha256"] == MODEL_SHA256
        # Fixed randomness is confined to this offline smoke process, never the server.
        with patch("secrets.randbits", rng.getrandbits):
            for mode in ("restricted", "free"):
                state = runtime.create(f"smoke-create-{mode}-00001",
                                       {"playMode": mode, "visibility": "developer"})
                session = state["sessionId"]
                for number in range(2):
                    state = runtime.new_hand(session, f"smoke-hand-{mode}-{number:016d}",
                                             {"revision": state["revision"]})
                    for step in range(100):
                        if state["phase"] == "finished":
                            break
                        body = {"revision": state["revision"], "handId": state["hand"]["id"]}
                        key = f"smoke-action-{mode}-{number:08d}-{step:08d}"
                        if state["hand"]["actor"] == 1:
                            state = runtime.advance(session, key, body)
                        else:
                            legal = state["hand"]["legal"]
                            if mode == "free" and number == 0 and step == 0:
                                body.update(kind="raise", raiseTo=201)
                            else:
                                body.update(kind="check" if "check" in legal["kinds"] else "call", raiseTo=None)
                            state = runtime.act(session, key, body)
                    assert state["phase"] == "finished"
                    assert runtime.verify_replay(session) == number + 1
                human_states.append(runtime.services["v0.4.2"]._load(session))
            for old in ("v0.4.1", "v0.4.0"):
                state = runtime.create(f"smoke-spectator-{old}-00001",
                                       {"sessionType": "spectator", "modelVersions": ["v0.4.2", old]})
                session = state["sessionId"]
                for number in range(2):
                    state = runtime.new_hand(session, f"smoke-spectator-hand-{old}-{number:08d}",
                                             {"revision": state["revision"]})
                    for step in range(100):
                        if state["phase"] == "finished":
                            break
                        state = runtime.advance(session, f"smoke-spectator-step-{old}-{number:08d}-{step:08d}",
                                                {"revision": state["revision"], "handId": state["hand"]["id"]})
                    assert state["phase"] == "finished"
                    assert runtime.verify_replay(session) == number + 1
        spectator_states = [json.loads(row[0]) for row in runtime.spectator.db.execute("SELECT state FROM sessions")]
        policies = {v: service.policy for v, service in runtime.services.items()}
        spectator = audit_spectator(spectator_states, policies, runtime.identities)
        web = policies["v0.4.2"]
    finally:
        runtime.close()
    (out / "human-states.json").write_text(json.dumps(human_states) + "\n")
    (out / "spectator-states.json").write_text(json.dumps(spectator_states) + "\n")
    (out / "spectator-audit.json").write_text(json.dumps(spectator, indent=2) + "\n")
    del runtime, policies
    gc.collect()
    reference = AveragePolicy(models / ASSET_NAME, MODEL_SHA256)
    human = audit_human(human_states, web, reference, MODEL_SHA256)
    assert human["counts"]["hands"] == 4 and human["counts"]["exact_201_chip_human_raises"] >= 1
    (out / "human-audit.json").write_text(json.dumps(human, indent=2) + "\n")
    result = {"status": "verified", "scope": "deterministic runtime smoke, no strength estimate",
              "source": source, "catalog": catalog, "human_counts": human["counts"],
              "human_positions_sha256": human["positions_sha256"], "spectator": spectator,
              "default_model_sha256": MODEL_SHA256}
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-version", required=True)
    args = parser.parse_args()
    print(json.dumps(smoke(args.models_dir, args.out, args.source_version), sort_keys=True))


if __name__ == "__main__":
    main()
