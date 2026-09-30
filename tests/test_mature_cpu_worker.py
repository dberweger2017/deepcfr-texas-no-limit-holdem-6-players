import os
from pathlib import Path

from scripts.mature_cpu_linux_worker import guard_reason, owned_rss, read_limit

LIMITS = {"max_owned_rss_gib": 10.5, "max_rss_fraction_of_container_memory": .8,
          "max_swap_growth_gib": .5, "min_free_disk_gib": 8}


def test_ram_guard_uses_actual_container_allocation():
    assert guard_reason(7*2**30, 8_000_000_000, 0, 20*2**30, 10, 20, LIMITS)
    assert guard_reason(5*2**30, 8_000_000_000, 0, 20*2**30, 10, 20, LIMITS) is None
    assert guard_reason(11*2**30, 64*2**30, 0, 20*2**30, 10, 20, LIMITS)


def test_swap_disk_and_immutable_deadline_are_separate_guards():
    assert guard_reason(1, 16*2**30, .5*2**30+1, 20*2**30, 10, 20, LIMITS) == "Swap growth guard"
    assert guard_reason(1, 16*2**30, 0, 8*2**30-1, 10, 20, LIMITS) == "Free disk guard"
    assert guard_reason(1, 16*2**30, 0, 20*2**30, 20, 20, LIMITS) == "Immutable rental/work deadline guard"


def test_host_unlimited_memory_is_not_treated_as_the_pod_allocation(tmp_path):
    path = tmp_path / "limit"
    path.write_text("max\n")
    assert read_limit((path,)) is None
    path.write_text(str(2**62))
    assert read_limit((path,)) is None
    path.write_text("8000000000\n")
    assert read_limit((path,)) == 8_000_000_000


def test_owned_memory_counts_all_descendants_without_foreign_host_jobs(tmp_path):
    def process(pid, parent, rss):
        p=tmp_path / str(pid)
        p.mkdir()
        (p/"status").write_text(f"Name:\tfixture\nPPid:\t{parent}\nVmRSS:\t{rss} kB\n")
    process(100, 1, 100)
    process(101, 100, 200)
    process(102, 101, 300)
    process(999, 1, 9000000)
    process(os.getpid(), 1, 50)
    assert owned_rss(100, tmp_path) == 650*1024


def test_monitor_failure_cleanup_escalates_only_our_process_group(monkeypatch):
    import signal
    import subprocess
    from scripts.mature_cpu_linux_worker import stop_child
    calls = []
    class Child:
        pid = 12345678
        def poll(self):
            return None
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout:
                raise subprocess.TimeoutExpired("fixture", timeout)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    stop_child(Child())
    assert calls == [(12345678, signal.SIGTERM), ("wait", 10),
                     (12345678, signal.SIGKILL), ("wait", None)]
