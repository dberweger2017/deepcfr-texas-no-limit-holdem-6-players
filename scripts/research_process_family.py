"""Discover owned descendants without launching a sampling helper process."""
import os
import psutil


def owned_processes(known, *, root=None, parent_rows=None, process_factory=psutil.Process):
    """Retain creation identities across sessions/reparenting; fail closed on access."""
    root = os.getpid() if root is None else root
    observed_creation = {}
    if parent_rows is None:
        snapshot = [p.info for p in psutil.process_iter(['pid', 'ppid', 'create_time'])]
        parent_rows = [(p['pid'], p['ppid']) for p in snapshot]
        observed_creation = {p['pid']: p['create_time'] for p in snapshot}
    retained = {}
    for pid, expected in list(known.items()):
        try:
            process = process_factory(pid)
            if process.create_time() == expected:
                retained[pid] = process
            else:
                del known[pid]
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            known.pop(pid, None)
    children = {}
    for pid, parent in parent_rows:
        children.setdefault(parent, []).append(pid)
    descendants, pending = set(), [root, *retained]
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
            # Iterator attributes may contain None after AccessDenied. Read an
            # owned identity afresh; never use the iterator's cached Process.
            observed = observed_creation.get(pid)
            if observed is not None and observed != created:
                raise RuntimeError('Owned PID changed during discovery: '+str(pid))
            if pid in known and known[pid] != created:
                del known[pid]
                if pid not in descendants:
                    continue
            known[pid] = created
            result.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            known.pop(pid, None)
    return result
