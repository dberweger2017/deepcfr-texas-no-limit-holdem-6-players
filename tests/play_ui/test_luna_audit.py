import json

from scripts.audit_luna_browser import audit


def call(namespace, name, code):
    return {"type": "response_item", "payload": {
        "type": "function_call", "namespace": namespace, "name": name,
        "call_id": "example", "arguments": json.dumps({"code": code})}}


def test_extracts_visible_metadata_without_reasoning_or_cards():
    records = [
        {"type": "turn_context", "payload": {"model": "gpt-6-luna", "effort": "high"}},
        {"type": "response_item", "payload": {"type": "reasoning", "text": "PRIVATE REASONING"}},
        call("mcp__cua_repl", "js", 'await table.click(12); await table.getAXState();'),
        {"type": "response_item", "payload": {"type": "function_call_output", "output":
            [{"type": "input_text", "text": '{"type":"luna_attempt","handOrdinal":1,"humanCards":["As","Ks"],"board":[],"attemptedButtonLabel":"check"}'}]}},
    ]
    result = audit(records)
    assert not result["violations"]
    assert result["configurations"] == [{"model": "gpt-6-luna", "effort": "high"}]
    assert result["decisionMetadata"] == [{"type": "luna_attempt", "handOrdinal": 1,
                                            "attemptedButtonLabel": "check", "lastRenderedAtMs": None}]
    assert "PRIVATE REASONING" not in json.dumps(result)
    assert "As" not in json.dumps(result)


def test_rejects_shell_browser_evaluation_and_other_origins():
    for record in (
        call("functions", "exec", "shell"),
        call("mcp__cua_repl", "js", 'await table.playwright.evaluate(() => state);'),
        call("mcp__cua_repl", "js", 'await table.goto("https://example.com/");'),
        call("mcp__cua_repl", "js", 'await cua.getState();'),
    ):
        assert audit([record])["violations"]
