"""Each backend marks the model's own prose `author: agent`; tool echoes and
connection notes carry no author. The review stores only the former."""
from __future__ import annotations

from backend.agents.claude import ClaudeBackend
from backend.agents.codex import CodexBackend
from backend.agents.kimi import KimiBackend


def _authored(events):
    return [(e["text"], e.get("author")) for e in events if e.get("type") == "log"]


def test_claude():
    evt = {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "VERDICT: PASS"},
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
    ]}}
    got = _authored(ClaudeBackend().parse_event(evt))
    assert got == [("VERDICT: PASS", "agent"), ("$ ls", None)]
    init = {"type": "system", "subtype": "init"}
    assert _authored(ClaudeBackend().parse_event(init)) == [
        ("Coding agent connected", None)]


def test_codex():
    msg = {"type": "item.completed",
           "item": {"id": "i1", "type": "agent_message", "text": "VERDICT: PASS"}}
    cmd = {"type": "item.completed",
           "item": {"id": "i2", "type": "command_execution", "command": "ls"}}
    assert _authored(CodexBackend().parse_event(msg)) == [("VERDICT: PASS", "agent")]
    assert _authored(CodexBackend().parse_event(cmd)) == [("$ ls", None)]


def test_kimi():
    evt = {"role": "assistant", "content": "VERDICT: PASS", "tool_calls": [
        {"function": {"name": "Shell", "arguments": '{"command": "ls"}'}}]}
    assert _authored(KimiBackend().parse_event(evt)) == [
        ("VERDICT: PASS", "agent"), ("$ ls", None)]
