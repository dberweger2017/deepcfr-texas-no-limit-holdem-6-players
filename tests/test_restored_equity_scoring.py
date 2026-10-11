"""The final qualification supersedes #222's retained preparation binary pin."""
import hashlib
import json
import pytest
from scripts import score_restored_hu20_equity_bench as scoring


def qualification(tmp_path, monkeypatch, *, final='scoring-binary', locator='scoring-binary'):
    original = {'lock': 'retained-earlier-binary'}
    (tmp_path/'input-pins.json').write_text(json.dumps(original))
    content = json.dumps({'lock_sha256': final}).encode()
    path = tmp_path/'qualified-final-source.json'
    path.write_bytes(content)
    root = tmp_path/'repo'
    index = root/'docs/reports/hu20-equity-bench-artifacts/model-input-index.json'
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({'lock_evaluator': {'sha256': locator}}))
    monkeypatch.setattr(scoring.bench, 'ROOT', root)
    return {'members': [{'path': 'research/qualified-final-source.json',
             'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}]}


def test_final_qualification_supersedes_retained_preparation(tmp_path, monkeypatch):
    receipt = qualification(tmp_path, monkeypatch)
    assert scoring.qualified_evaluator(tmp_path, receipt) == 'scoring-binary'
    assert json.loads((tmp_path/'input-pins.json').read_text())['lock'] == 'retained-earlier-binary'


def test_final_qualification_must_match_published_locator(tmp_path, monkeypatch):
    receipt = qualification(tmp_path, monkeypatch, locator='other-binary')
    with pytest.raises(ValueError, match='Final qualification and restoration locator differ'):
        scoring.qualified_evaluator(tmp_path, receipt)


def test_qualification_member_cannot_be_rewritten(tmp_path, monkeypatch):
    receipt = qualification(tmp_path, monkeypatch)
    (tmp_path/'qualified-final-source.json').write_text('{}')
    with pytest.raises(ValueError, match='Archived evaluator qualification differs'):
        scoring.qualified_evaluator(tmp_path, receipt)
