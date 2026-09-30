"""Sample one owned macOS service and system memory without recording command arguments."""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    with args.output.open("a", buffering=1) as stream:
        while True:
            try:
                process = command(["ps", "-p", str(args.pid), "-o", "rss=,%cpu=,etime="]).split()
            except subprocess.CalledProcessError:
                break
            if not process:
                break
            sample = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "serviceRssKiB": int(process[0]), "serviceCpuPercent": float(process[1]),
                      "serviceElapsed": process[2],
                      "swap": command(["sysctl", "vm.swapusage"]),
                      "vmStat": command(["vm_stat"]),
                      "top": command(["top", "-pid", str(args.pid), "-l", "1",
                                      "-stats", "pid,command,cpu,mem,rprvt,vsize"])}
            stream.write(json.dumps(sample) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
