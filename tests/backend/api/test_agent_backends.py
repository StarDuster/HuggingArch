"""`/api/agents`: the agent backends pages offer and label, from the one
registry (`backend.agents`), not from a list each page keeps."""
from fastapi.testclient import TestClient

from backend.agents import available_clis, get_backend
from backend.api import app


def test_agents_are_the_registry_with_their_display_names(monkeypatch):
    monkeypatch.delenv("SPEC_AGENT_CLI", raising=False)
    body = TestClient(app).get("/api/agents").json()
    assert [a["cli"] for a in body["agents"]] == list(available_clis())
    for agent in body["agents"]:
        assert agent["label"] == get_backend(agent["cli"]).display_name
    assert body["default"] in available_clis()


def test_default_is_the_configured_agent(monkeypatch):
    other = available_clis()[-1]
    monkeypatch.setenv("SPEC_AGENT_CLI", other)
    assert TestClient(app).get("/api/agents").json()["default"] == other
