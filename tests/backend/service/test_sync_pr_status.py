"""Poll-on-view reconciliation of a spec_generation's PR state.

A contributed record whose PR closed without merging gets a terminal `closed`
status (distinct from admin `rejected`); a merged PR gets `merged`; a still-open
PR is left untouched. No webhook — this runs when a record is viewed.
"""
from backend.service import spec_generations as G


def _spy_set_status(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "set_status", lambda rid, status, **kw: (calls.append((rid, status)) or True))
    return calls


def test_closed_pr_flips_contributed_to_closed(monkeypatch):
    calls = _spy_set_status(monkeypatch)
    import backend.service.github as gh
    monkeypatch.setattr(gh, "pr_merged_state", lambda url: "closed")
    rec = {"id": "r1", "status": "contributed", "pr_url": "https://x/pull/1"}
    assert G.sync_pr_status(rec) == "closed"
    assert calls == [("r1", "closed")]


def test_merged_pr_flips_to_merged(monkeypatch):
    calls = _spy_set_status(monkeypatch)
    import backend.service.github as gh
    monkeypatch.setattr(gh, "pr_merged_state", lambda url: "merged")
    rec = {"id": "r2", "status": "contributed", "pr_url": "https://x/pull/2"}
    assert G.sync_pr_status(rec) == "merged"
    assert calls == [("r2", "merged")]


def test_open_pr_is_left_untouched(monkeypatch):
    calls = _spy_set_status(monkeypatch)
    import backend.service.github as gh
    monkeypatch.setattr(gh, "pr_merged_state", lambda url: "open")
    rec = {"id": "r3", "status": "contributed", "pr_url": "https://x/pull/3"}
    assert G.sync_pr_status(rec) is None
    assert calls == []


def test_already_merged_is_not_polled(monkeypatch):
    calls = _spy_set_status(monkeypatch)
    import backend.service.github as gh
    called = []
    monkeypatch.setattr(gh, "pr_merged_state", lambda url: called.append(url) or "closed")
    rec = {"id": "r4", "status": "merged", "pr_url": "https://x/pull/4"}
    assert G.sync_pr_status(rec) is None
    assert called == []  # merged is terminal — no GitHub call, no downgrade
    assert calls == []


def test_no_pr_url_is_noop(monkeypatch):
    calls = _spy_set_status(monkeypatch)
    rec = {"id": "r5", "status": "generated", "pr_url": None}
    assert G.sync_pr_status(rec) is None
    assert calls == []


def test_pr_merged_state_hits_the_url_repo(monkeypatch):
    import backend.service.github as gh
    monkeypatch.setattr(gh, "enabled", lambda: True)
    monkeypatch.setattr(gh, "_headers", lambda: {"Authorization": "Bearer x"})
    seen = []

    class _Resp:
        status_code = 200
        def json(self):
            return {"merged": True, "state": "closed"}

    class _Cx:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url):
            seen.append(url)
            return _Resp()

    monkeypatch.setattr(gh.httpx, "Client", _Cx)
    assert gh.pr_merged_state("https://github.com/acme/fork/pull/99") == "merged"
    assert seen == ["https://api.github.com/repos/acme/fork/pulls/99"]


def test_rejected_is_terminal_even_with_a_pr_url(monkeypatch):
    # An admin's rejection is terminal; the PR reconciler must never override it
    # (a record can carry a pr_url yet still have been rejected afterward).
    calls = _spy_set_status(monkeypatch)
    import backend.service.github as gh
    called = []
    monkeypatch.setattr(gh, "pr_merged_state", lambda url: called.append(url) or "merged")
    rec = {"id": "r6", "status": "rejected", "pr_url": "https://x/pull/6"}
    assert G.sync_pr_status(rec) is None
    assert called == []   # short-circuits before the GitHub call
    assert calls == []
