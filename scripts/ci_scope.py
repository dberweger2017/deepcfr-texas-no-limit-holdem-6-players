"""Route prose-only PRs without weakening main or unknown-change validation."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{40}")


def git(*args):
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)


def changes(base, head):
    parts = git("diff", "--name-status", "-z", "--find-renames", base, head).split(b"\0")
    rows = []
    i = 0
    while i < len(parts) - 1:
        status = parts[i].decode("ascii")
        i += 1
        count = 2 if status.startswith(("R", "C")) else 1
        paths = [parts[i + n].decode("utf-8") for n in range(count)]
        rows.append((status, paths))
        i += count
    return rows


def inspect(event_name, event, head):
    if event_name != "pull_request":
        return True, "Main pushes and manual runs always use full validation", []
    base = event.get("pull_request", {}).get("base", {}).get("sha", "")
    if not isinstance(base, str) or not SHA.fullmatch(base) or not SHA.fullmatch(head):
        return True, "Missing or invalid comparison revision", []
    rows = changes(base, head)
    if not rows:
        return True, "No changes available to classify", []
    prose = []
    for status, paths in rows:
        if status not in ("A", "M"):
            return True, "Deleted, renamed or changed file type requires full validation", []
        path = paths[0]
        if not path.endswith(".md") or path == "docs/rules.md":
            return True, "Changes include code, contracts, configuration or data", []
        for revision in ([base, head] if status == "M" else [head]):
            entries = git("ls-tree", "-z", revision, "--", path).split(b"\0")[:-1]
            if (len(entries) != 1 or not entries[0].startswith(b"100644 blob ")
                    or entries[0].split(b"\t", 1)[1].decode("utf-8") != path):
                return True, "Only ordinary tracked prose files qualify", []
        prose.append(path)
    return False, "Only added or edited Markdown prose", prose


def check_prose(head, paths):
    for path in paths:
        # Read tracked blobs, never filesystem symlinks or ignored private files.
        git("show", f"{head}:{path}").decode("utf-8")


def route(event_name, event, head):
    try:
        return inspect(event_name, event, head)
    except (KeyError, IndexError, ValueError, TypeError, AttributeError, subprocess.CalledProcessError):
        return True, "Unable to classify safely; using full validation", []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-prose", action="store_true")
    args = parser.parse_args()
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        head = git("rev-parse", "HEAD").decode().strip()
        full, reason, paths = route(os.environ.get("GITHUB_EVENT_NAME", ""), event, head)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError):
        full, reason, paths = True, "Missing event or repository context; using full validation", []
    if args.check_prose:
        if full:
            raise SystemExit("Prose check requires a verified prose-only comparison")
        check_prose(head, paths)
    if not args.check_prose:
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a") as stream:
                stream.write(f"full={'true' if full else 'false'}\n")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as stream:
                stream.write(f"## CI scope\n\nLane: **{'full' if full else 'prose'}**. {reason}.\n")
    print(json.dumps({"full": full, "reason": reason, "proseFiles": len(paths)}))


if __name__ == "__main__":
    main()
