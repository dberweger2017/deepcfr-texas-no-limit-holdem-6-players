"""The released table can run without neural or campaign modules installed."""

import subprocess
import sys

from src.diagnostics.cfr_average import extract
from tests.diagnostics.test_cfr_average import fixture


def test_play_and_average_inference_do_not_import_research_tools(tmp_path):
    _, _, checkpoint, _, spec = fixture(tmp_path)
    exported = tmp_path / 'average.gz'
    receipt = extract(checkpoint, spec, exported)
    code = '''
import importlib.abc
import sys
from pathlib import Path
class ResearchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in
               ('torch', 'scipy', 'scripts', 'src.diagnostics')):
            raise ImportError('Play imported research dependency: ' + fullname)
sys.meta_path.insert(0, ResearchBlocker())
from src.play_api import server, versions, shield
from src.blueprint.average import AveragePolicy
from src.game.hand import Hand, Table
policy = AveragePolicy(Path(sys.argv[1]), sys.argv[2])
view = Hand.start(Table(('a', 'b'), (2000, 2000)), hand_id='runtime', seed=17).observe(0)
action = policy.policy(13).choose_action(view)
view.legal_actions.validate(action)
assert policy.distribution(view)[1][0] == 0
assert versions.DEFAULT_VERSION == 'v0.4.1'
'''
    subprocess.run([sys.executable, '-c', code, str(exported), receipt['sha256']], check=True)


def test_download_verifiers_work_before_engine_installation():
    for module in ('scripts.verify_v04_model', 'scripts.verify_v041_model'):
        subprocess.run([sys.executable, '-S', '-m', module, '--help'],
                       check=True, capture_output=True)
