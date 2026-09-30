"""Summarize macOS resource samples separately from poker outcomes."""

import argparse
import csv
import json
import re
from pathlib import Path


def summarize(samples):
    rows = []
    for sample in samples:
        top = sample["top"]
        process = next(line.split() for line in reversed(top.splitlines())
                       if re.match(r"^\d+\s+Python\s", line))
        rows.append({"timestamp": sample["timestamp"],
                     "serviceRssKiB": sample["serviceRssKiB"],
                     "serviceCpuPercent": sample["serviceCpuPercent"],
                     "serviceTopMem": process[3],
                     "hostCpuIdlePercent": float(re.search(r"([\d.]+)% idle", top)[1]),
                     "hostCompressor": re.search(r"([\d.]+[KMG]) compressor", top)[1],
                     "hostSwapUsedMiB": float(re.search(r"used = ([\d.]+)M", sample["swap"])[1])})
    summary = {"samples": len(rows), "firstSample": rows[0]["timestamp"],
               "lastSample": rows[-1]["timestamp"],
               "serviceRssKiB": {"min": min(r["serviceRssKiB"] for r in rows),
                                  "max": max(r["serviceRssKiB"] for r in rows)},
               "serviceTopMemValues": sorted({r["serviceTopMem"] for r in rows}),
               "serviceCpuPercentMaxObserved": max(r["serviceCpuPercent"] for r in rows),
               "hostSwapUsedMiB": {"min": min(r["hostSwapUsedMiB"] for r in rows),
                                    "max": max(r["hostSwapUsedMiB"] for r in rows)},
               "scope": "30-second samples after startup; observed maxima, not guaranteed lifetime peaks",
               "memoryInterpretation": "ps RSS and top MEM are distinct; retain top display units. Compression and swap are host-wide, include unrelated applications, and cannot be attributed wholly to the experiment."}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows, summary = summarize([json.loads(line) for line in args.samples.open()])
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "resources.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "resources.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
