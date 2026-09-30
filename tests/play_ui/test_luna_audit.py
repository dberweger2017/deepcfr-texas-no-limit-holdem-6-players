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
            [{"type": "input_text", "text": '{"type":"luna_attempt","handOrdinal":1,"humanCards":["As","Ks"],"board":[],"visibleHumanCards":["As","Ks"],"visibleBoard":[],"attemptedButtonLabel":"check"}'}]}},
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


def test_metadata_key_order_does_not_drop_attempts():
    record = {"type": "response_item", "payload": {"type": "function_call_output", "output":
              '{"handOrdinal":1,"attemptedButtonLabel":"fold","type":"luna_attempt"}'}}
    assert len(audit([record])["decisionMetadata"]) == 1


def test_counts_expired_browser_handle_without_copying_session_identifier():
    record = {"type": "response_item", "payload": {"type": "function_call_output",
              "call_id": "expired", "output": "Tab 3 is not part of browser session PRIVATE-ID"}}
    result = audit([record])
    assert result["setupFailures"] == [{"callId": "expired", "category": "expired browser handle"}]
    assert "PRIVATE-ID" not in json.dumps(result)


def test_normalizes_observable_ordinal_spellings_without_copying_rendered_tree():
    record = {"type": "response_item", "payload": {"type": "function_call_output", "output":
              json.dumps({"type": "luna_observed", "hand": 400, "decision": 1,
                          "acceptedAction": "Call 2 BB", "visibleState": "FULL RENDERED TREE",
                          "state": "FULL RENDERED TREE", "arbitraryPrivateField": "PRIVATE"})}}
    item = audit([record])["decisionMetadata"][0]
    assert item["handOrdinal"] == item["hand"] == 400
    assert item["decisionOrdinal"] == item["decision"] == 1
    assert item["visibleAcceptedAction"] == "Call 2 BB"
    assert "FULL RENDERED TREE" not in json.dumps(item)
    assert "PRIVATE" not in json.dumps(item)
