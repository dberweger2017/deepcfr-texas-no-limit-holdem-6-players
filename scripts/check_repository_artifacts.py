"""Keep new research payloads out of Git; inventory retained evidence without deleting it."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import subprocess

MAX_BYTES = 1024 * 1024
POLICY_PATH = 'configs/repository-artifacts.json'
LOCAL_ROOTS = {'models', 'results', 'planning', 'logs'}
PAYLOAD_SUFFIXES = {
    '.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.npz', '.npy', '.pkl',
    '.pickle', '.bin', '.zip', '.gz', '.tgz', '.bz2', '.xz', '.zst', '.tar',
    '.7z', '.jsonl', '.sqlite', '.sqlite3', '.db',
}


@dataclass(frozen=True)
class Entry:
    path: str
    blob: str
    size: int


def git(*args, input=None):
    return subprocess.check_output(['git', *args], input=input)


def entries(revision=None):
    """Inspect Git objects, not local working files or symlink targets."""
    objects = []
    if revision is not None:
        revision = git('rev-parse', '--verify', '--end-of-options', revision + '^{tree}').decode().strip()
        for row in git('ls-tree', '-r', '-z', revision).split(b'\0'):
            if not row:
                continue
            meta, path = row.split(b'\t', 1)
            _, kind, blob = meta.decode().split()
            if kind != 'blob':
                raise ValueError('Submodules need an explicit storage review')
            objects.append((path.decode(), blob))
    else:
        for row in git('ls-files', '--stage', '-z').split(b'\0'):
            if not row:
                continue
            meta, path = row.split(b'\t', 1)
            mode, blob, stage = meta.decode().split()
            if stage != '0' or mode == '160000':
                raise ValueError('Resolve unmerged entries/submodules before checking storage')
            objects.append((path.decode(), blob))
    if not objects:
        return []
    sizes = git('cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)',
                input=''.join(blob + '\n' for _, blob in objects).encode()).decode().splitlines()
    output = []
    for (path, blob), row in zip(objects, sizes, strict=True):
        actual, kind, size = row.split()
        if actual != blob or kind != 'blob':
            raise ValueError(f'Cannot inspect tracked blob: {path}')
        output.append(Entry(path, blob, int(size)))
    return output


def reasons(entry):
    path = PurePosixPath(entry.path)
    found = []
    if path.parts[0] in LOCAL_ROOTS:
        found.append('local research directory')
    if {suffix.lower() for suffix in path.suffixes} & PAYLOAD_SUFFIXES or 'tfevents' in path.name.lower():
        found.append('model, raw trace, database or archive')
    if entry.size > MAX_BYTES:
        found.append('larger than 1 MiB')
    return found


def violations(tracked, policy):
    if policy.get('version') != 1:
        raise ValueError('Unknown repository artifact policy version')
    exceptions = policy['retained']
    problems = []
    for entry in tracked:
        why = reasons(entry)
        if not why:
            continue
        allowed = exceptions.get(entry.path, {})
        if (allowed.get('git_blob') == entry.blob and allowed.get('bytes') == entry.size
                and allowed.get('reason')):
            continue
        problems.append(f'{entry.path}: {", ".join(why)}; store in Research-Cloud and index retrieval')
    return problems


def inventory(tracked):
    counts, sizes = Counter(), Counter()
    retained = []
    for entry in tracked:
        root = entry.path.split('/')[0]
        counts[root] += 1
        sizes[root] += entry.size
        if why := reasons(entry):
            retained.append({'path': entry.path, 'git_blob': entry.blob,
                             'bytes': entry.size, 'reasons': why})
    return {'tracked_files': len(tracked), 'tracked_bytes': sum(sizes.values()),
            'directories': {root: {'files': counts[root], 'bytes': sizes[root]} for root in sorted(counts)},
            'research_payload_candidates': retained,
            'scope': 'Git inventory only; neither Drive upload nor deletion eligibility is verified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', help='Inspect a committed tree instead of the staged index')
    parser.add_argument('--inventory', action='store_true', help='Print JSON inventory; never mutate files')
    args = parser.parse_args()
    try:
        tracked = entries(args.revision)
        if args.inventory:
            print(json.dumps(inventory(tracked), indent=2))
            return 0
        # The policy comes from the same Git snapshot as the files being checked.
        policy_entry = next((item for item in tracked if item.path == POLICY_PATH), None)
        if policy_entry is None:
            raise ValueError(f'Missing tracked policy: {POLICY_PATH}')
        policy = json.loads(git('cat-file', 'blob', policy_entry.blob))
        errors = violations(tracked, policy)
        if errors:
            print('\n'.join(errors))
            return 1
        print(f'Storage check passed: {len(tracked)} tracked files; legacy payloads pinned by exact Git blob.')
        return 0
    except (ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Storage check failed: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
