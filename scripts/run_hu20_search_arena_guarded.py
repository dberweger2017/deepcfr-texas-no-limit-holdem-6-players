"""Add operational admission to the unchanged approved arena worker."""

from collections import Counter
import os
from time import monotonic

from scripts.hu20_search_arena_control import ControlClient


def guarded_policy(policy_class, client, guard):
    class GuardedPolicy(policy_class):
        def distribution(self, view, *, query_kind="probe"):
            if query_kind != "play" or view.street.value not in ("turn", "river"):
                return super().distribution(view, query_kind=query_kind)
            event = client.acquire(guard)
            before = Counter(self.stats)
            result = super().distribution(view, query_kind=query_kind)
            delta = self.stats - before
            causes = [key.split("play:fallback:", 1)[1] for key, count in delta.items()
                      if key.startswith("play:fallback:") and count]
            gap = any(k.startswith("range:turn_conditioning_fallback:") and v for k, v in delta.items())
            if gap or any(c not in ("timeout", "memory_refusal", "unsupported_holding", "zero_support") for c in causes):
                client.request("stop", reason="Correctness/conditioning guard: " + str(causes))
                raise RuntimeError("Correctness/conditioning guard")
            client.request("complete", event=event, fallback=bool(causes), cause=",".join(sorted(causes)) or None)
            return result
    return GuardedPolicy


def main():
    from scripts import evaluate_hu20_turn_search as worker
    client = ControlClient(os.environ["HU20_CONTROL_URL"], os.environ["HU20_POD_ID"],
                           int(os.environ["HU20_WORKER_INDEX"]), os.environ.get("HU20_CONTROL_TOKEN"))
    from scripts import hu20_search_runtime as runtime
    # These community hosts mix v1/v2. Do not charge unrelated host swap to the
    # pod when a v1 memory controller exposes its own memsw counters.
    from pathlib import Path
    original_swap = runtime.swap_bytes
    def pod_swap():
        base = Path("/sys/fs/cgroup/memory")
        total, resident = base/"memory.memsw.usage_in_bytes", base/"memory.usage_in_bytes"
        if total.exists() and resident.exists():
            return max(0, int(total.read_text())-int(resident.read_text()))
        return original_swap()
    runtime.swap_bytes = pod_swap
    original_budget = worker.PaidWorkerBudget
    active = []

    class GuardedBudget(original_budget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.control_check_at = 0
            active.append(self)

        def check(self):
            super().check()
            if monotonic() - self.control_check_at >= 1:
                client.request("check")
                self.control_check_at = monotonic()

    worker.PaidWorkerBudget = GuardedBudget
    worker.HU20TurnSearchPolicy = guarded_policy(worker.HU20TurnSearchPolicy, client,
                                                 lambda: active[0].check())
    try:
        worker.main()
    except BaseException:
        try:
            client.request("stop", reason=f"Worker {client.worker} exited incompletely; retain partials")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
