"""Exercise the actual browser playback controller's in-flight request races."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_playback_request_order_and_pause():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the playback controller checks')
    root = Path(__file__).resolve().parents[2]
    subprocess.run([node, '--test', str(Path(__file__).with_name('playback.test.cjs'))],
                   cwd=root, check=True, capture_output=True, text=True)
