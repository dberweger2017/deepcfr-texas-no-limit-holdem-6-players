"""Add operational admission to the unchanged approved arena worker."""

from collections import Counter
import json
import os
from time import monotonic, time

from scripts.hu20_search_arena_control import ControlClient


def guarded_policy(policy_class, client, guard, defect_log=None):
    """Owner rule for the autonomous arena: record every fallback/defect per decision and keep playing.

    The policy falls back to its base action with the cause and timing retained, and later queries in
    the hand stay consistent with that fallback. A process crash is handled by the pod's worker loop.
    """
    class GuardedPolicy(policy_class):
        continue_on_defect = True

        def distribution(self, view, *, query_kind="probe"):
            if query_kind != "play" or view.street.value not in ("turn", "river"):
                return super().distribution(view, query_kind=query_kind)
            event = client.acquire(guard)
            before = Counter(self.stats)
            records_before = len(getattr(self, "records", ()))
            result = super().distribution(view, query_kind=query_kind)
            delta = self.stats - before
            causes = [key.split("play:fallback:", 1)[1] for key, count in delta.items()
                      if key.startswith("play:fallback:") and count]
            gaps = [k for k, v in delta.items() if k.startswith("range:turn_conditioning_fallback:") and v]
            if (causes or gaps) and defect_log is not None:
                with open(defect_log, "a") as stream:
                    stream.write(json.dumps({
                        "at": time(), "pod": client.pod, "worker": client.worker, "event": event,
                        "hand_id": view.hand_id, "street": view.street.value, "causes": sorted(causes),
                        "conditioning_gaps": sorted(gaps),
                        "records": [{k: r.get(k) for k in ("status", "cause", "query_kind", "seconds")}
                                    for r in getattr(self, "records", [])[records_before:]]}, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            client.request("complete", event=event, fallback=bool(causes or gaps),
                           cause=",".join(sorted(causes + gaps)) or None)
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
                                                 lambda: active[0].check(),
                                                 os.environ.get("HU20_DEFECT_LOG"))
    worker.main()


if __name__ == "__main__":
    main()
