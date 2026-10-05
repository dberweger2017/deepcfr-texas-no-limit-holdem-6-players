"""Read-only status/TensorBoard sidecar; never launches or changes experiment work."""

import argparse
import json
from pathlib import Path
import time


def main():
    from torch.utils.tensorboard import SummaryWriter
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True);p.add_argument("--logdir",type=Path,required=True)
    a=p.parse_args()
    last=None
    with SummaryWriter(str(a.logdir)) as writer:
        while True:
            path=a.run/"status.json"
            if path.exists():
                data=json.loads(path.read_text());stamp=path.stat().st_mtime_ns
                if stamp != last:
                    last=stamp;step=data.get("hands",data.get("roots",0))
                    for key in ("hands","roots","expected_hands","seconds","fallbacks","peak_owned_rss_bytes"):
                        if isinstance(data.get(key),(int,float)):
                            writer.add_scalar("progress/"+key,data[key],step)
                    writer.add_text("status",json.dumps(data,sort_keys=True),step);writer.flush()
                if data.get("status") in ("complete","incomplete","failed","qualified","owner-decision-needed"):
                    break
            time.sleep(5)


if __name__ == "__main__":main()
