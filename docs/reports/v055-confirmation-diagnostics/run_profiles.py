"""Apply scripts.profile_slumbot_runs to every verified run, including the 30,000-hand pool."""
import json
from pathlib import Path
import sys

from scripts.profile_slumbot_runs import main

if __name__ == '__main__':
    runs = json.loads(Path('results/outputs/runs.json').read_text())
    sys.argv = ['profile_slumbot_runs']
    for name, folders in runs.items():
        sys.argv.extend(['--player', name + '=' + ','.join(folders)])
    sys.argv.extend(['--out', 'results/outputs/profiles.json',
                     '--markdown', 'results/outputs/profiles.md'])
    main()
