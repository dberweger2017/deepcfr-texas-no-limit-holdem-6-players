"""Extract observable tool metadata without copying private reasoning."""

import argparse
import hashlib
import json
import re
from datetime import datetime
from collections import Counter
from pathlib import Path

ORIGIN = "http://127.0.0.1:8765/"
FORBIDDEN_CODE = re.compile(
    r"\.(?:evaluate|evaluateAll|getAttribute|textContent|allTextContents)\s*\("
    r"|\.(?:dev|clipboard|content)\b|capabilities|\b(?:fetch|XMLHttpRequest|require|import)\b"
    r"|cua\.(?:getState|listApps|listWindows|listBrowsers|listTabs|getApp)\s*\("
)


def _metadata(output):
    if isinstance(output, list):
        for item in output:
            yield from _metadata(item)
        return
    if isinstance(output, dict):
        if isinstance(output.get("text"), str):
            yield from _metadata(output["text"])
        return
    if not isinstance(output, str):
        return
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{\s*"type"\s*:\s*"luna_(?:attempt|observed)"', output):
        try:
            record, _ = decoder.raw_decode(output[match.start():])
        except ValueError:
            continue
        # Cards can be checked privately against rendered observations, but are
        # not required in the published decision/tool metadata.
        yield {k: v for k, v in record.items() if k not in ("humanCards", "board")}


def audit(records):
    calls = []
    metadata = []
    violations = []
    configurations = []
    usages = []
    last_rendered_ms = None
    for record in records:
        kind = record.get("type")
        payload = record.get("payload", {})
        if kind == "turn_context":
            config = {"model": payload.get("model"), "effort": payload.get("effort")}
            if config not in configurations:
                configurations.append(config)
            continue
        if kind == "token_usage_record":
            usages.append({k: payload[k] for k in ("usage", "turn_token_usage", "thread_token_usage")
                           if k in payload})
            continue
        if kind != "response_item":
            continue
        if payload.get("type") in ("function_call", "custom_tool_call"):
            name = f'{payload.get("namespace", "")}.{payload.get("name", "")}'
            call = {"name": name, "callId": payload.get("call_id"),
                    "timestamp": record.get("timestamp")}
            calls.append(call)
            if name != "mcp__cua_repl.js":
                violations.append({**call, "reason": "prohibited tool"})
                continue
            try:
                code = json.loads(payload.get("arguments", "{}"))["code"]
            except (ValueError, KeyError, TypeError):
                violations.append({**call, "reason": "unreadable CUA invocation"})
                continue
            if FORBIDDEN_CODE.search(code):
                violations.append({**call, "reason": "prohibited browser/code capability"})
            urls = re.findall(r'https?://[^\s"\'<>]+', code)
            if any(url != ORIGIN for url in urls):
                violations.append({**call, "reason": "navigation outside table origin"})
            # Publish hashes, not tool code or the surrounding model transcript.
            call["codeSha256"] = hashlib.sha256(code.encode()).hexdigest()
        elif payload.get("type") in ("function_call_output", "custom_tool_call_output"):
            output = payload.get("output", "")
            extracted = list(_metadata(output))
            for item in extracted:
                if item["type"] == "luna_attempt":
                    item["lastRenderedAtMs"] = last_rendered_ms
            metadata.extend(extracted)
            # The result receipt bounds when this rendered observation reached
            # the player. It is not the browser's first-ready timestamp.
            if "Browser tab:" in json.dumps(output) and record.get("timestamp"):
                last_rendered_ms = int(datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")).timestamp() * 1000)
    return {"configurations": configurations, "toolCalls": calls,
            "toolCounts": dict(Counter(x["name"] for x in calls)),
            "violations": violations, "decisionMetadata": metadata,
            "usageRecords": usages}


def read(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(read(args.rollout))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"configurations": result["configurations"],
                      "toolCounts": result["toolCounts"],
                      "violations": result["violations"],
                      "metadataRows": len(result["decisionMetadata"])}, sort_keys=True))
    return int(bool(result["violations"]))


if __name__ == "__main__":
    raise SystemExit(main())
