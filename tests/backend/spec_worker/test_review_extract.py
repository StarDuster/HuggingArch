"""Verdict extraction from the review agent's streamed text.

The verdict is scanned out of the accumulated stream, not written to a file, so
the parser is the load-bearing part: it must find the sentinel-wrapped block,
fall back to a bare `VERDICT:` line, and default to `error` when the agent
produced no parseable verdict (which is still persisted).
"""
from backend.spec_worker.review_evidence import ReviewEvidence, SnapshotFacts
from backend.spec_worker.reviewer import _extract, _compose_report, _final_verdict, _verdict_note

# A snapshot with one source line, and a PASS report whose one Table F row
# quotes it (and whose Table S maps the node back): the evidence check has nothing to hold against the verdict.
_FACTS = SnapshotFacts({}, None, {"modeling_m.py": "h = self.norm(x)\n"})
_CLEAN = ReviewEvidence.audit(
    "| # | forward step | source | spec node(s) | tags | status |\n"
    "|---|---|---|---|---|---|\n"
    '| 1 | norm | `src: modeling_m.py:1 "h = self.norm(x)"` | norm | [sem] | match |\n'
    "\n| component / node | op | forward step # |\n|---|---|---|\n| norm | rmsnorm | 1 |\n",
    _FACTS)


def test_sentinel_block_pass():
    text = (
        "some tool logs...\n"
        "===REVIEW REPORT START===\n"
        "- attention: GQA, matches modeling source\n"
        "- moe: top-8, shared expert present\n"
        "VERDICT: PASS\n"
        "===REVIEW REPORT END===\n"
    )
    verdict, body = _extract(text)
    assert verdict == "pass"
    assert "attention: GQA" in body


def test_sentinel_block_no_pass():
    text = (
        "===REVIEW REPORT START===\n"
        "- rope: spec uses full rope but model is partial\n"
        "VERDICT: NO_PASS\n"
        "===REVIEW REPORT END===\n"
    )
    verdict, body = _extract(text)
    assert verdict == "no_pass"
    assert "partial" in body


def test_case_insensitive():
    verdict, _ = _extract("===review report start===\nx\nverdict: pass\n===review report end===")
    assert verdict == "pass"


def test_bare_verdict_fallback():
    # Agent forgot the sentinels but emitted the verdict line.
    verdict, body = _extract("I reviewed it.\nVERDICT: NO_PASS\nreasons above")
    assert verdict == "no_pass"
    assert "reviewed" in body


def test_no_verdict_is_error():
    verdict, body = _extract("the agent rambled and never concluded")
    assert verdict == "error"
    assert "rambled" in body


def test_empty_is_error():
    assert _extract("")[0] == "error"


def test_last_block_wins_over_earlier_draft():
    # An early draft block says PASS; the real (last) block says NO_PASS.
    text = (
        "===REVIEW REPORT START===\ndraft\nVERDICT: PASS\n===REVIEW REPORT END===\n"
        "on reflection the rope is wrong\n"
        "===REVIEW REPORT START===\n- rope wrong\nVERDICT: NO_PASS\n===REVIEW REPORT END===\n"
    )
    assert _extract(text)[0] == "no_pass"


def test_echoed_example_does_not_mis_scan():
    # A bare 'VERDICT: PASS' appears early (e.g. echoed), real conclusion last.
    text = "the format is VERDICT: PASS ... later:\nVERDICT: NO_PASS"
    assert _extract(text)[0] == "no_pass"


def test_final_verdict_pass_requires_valid_true_and_agent_pass():
    assert _final_verdict("pass", {"valid": True}, _CLEAN) == "pass"


def test_final_verdict_valid_false_forces_no_pass():
    assert _final_verdict("pass", {"valid": False}, _CLEAN) == "no_pass"
    assert _final_verdict("no_pass", {"valid": True}, _CLEAN) == "no_pass"


def test_final_verdict_gate_absent_cannot_pass():
    # validate() could not run (None) → agent PASS can't stand.
    assert _final_verdict("pass", None, _CLEAN) == "error"
    # a non-pass agent verdict with no gate is preserved.
    assert _final_verdict("no_pass", None, _CLEAN) == "no_pass"
    assert _final_verdict("error", None, _CLEAN) == "error"


def test_compose_report_prepends_validator_block():
    vr = {"valid": False, "errors": ["boom"]}
    out = _compose_report(vr, "agent findings here", _CLEAN, "no_pass")
    assert "Deterministic validator" in out
    assert "## Evidence check" in out
    assert "valid: False" in out
    assert "agent findings here" in out


def test_verdict_note_says_when_the_recorded_verdict_overrules_the_agent():
    """The log ends on the agent's own VERDICT line; its closing note names
    the verdict recorded, and why when the checks overruled the agent."""
    assert _verdict_note("pass", "pass", {"valid": True}, _CLEAN) == "[verdict] recorded PASS\n"
    no_facts = ReviewEvidence.audit("no tables", SnapshotFacts.empty())
    note = _verdict_note("error", "pass", {"valid": True}, no_facts)
    assert note.startswith("[verdict] recorded ERROR; the agent wrote PASS")
    assert "Table F" in note
    assert "validator reports errors" in _verdict_note("no_pass", "pass", {"valid": False}, _CLEAN)
