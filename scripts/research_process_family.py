"""Discover only owned macOS descendants without psutil's global PPID scan."""
import os
import subprocess
import psutil


def owned_processes(known, *, root=None, parent_rows=None, process_factory=psutil.Process):
    """Retain creation identities across sessions/reparenting; fail closed on access."""
    root = os.getpid() if root is None else root
    if parent_rows is None:
        text = subprocess.check_output(['ps', '-axo', 'pid=,ppid='], text=True, timeout=10)
        parent_rows = [tuple(map(int, line.split())) for line in text.splitlines() if line.strip()]
    children = {}
    for pid, parent in parent_rows:
        children.setdefault(parent, []).append(pid)
    descendants, pending = set(), [root]
    while pending:
        parent = pending.pop()
        for pid in children.get(parent, ()):
            if pid != root and pid not in descendants:
                descendants.add(pid)
                pending.append(pid)
    result = [process_factory(root)]
    for pid in sorted(set(known) | descendants):
        try:
            process = process_factory(pid)
            created = process.create_time()
            if pid in known and known[pid] != created:
                del known[pid]
                if pid not in descendants:
                    continue
            known[pid] = created
            result.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            known.pop(pid, None)
    return result
