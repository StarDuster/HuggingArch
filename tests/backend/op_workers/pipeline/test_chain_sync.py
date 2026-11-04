"""run_propose_review is a sync propose→review with no live-tree splice."""
from __future__ import annotations

import json

from backend.op_workers.pipeline import chain as chain_mod


def _ok_done():
    yield 'data: {"type": "log", "text": "ok"}\n\n'
    yield 'data: {"type": "done", "success": true}\n\n'


def test_ready_when_reviewer_auto_merges(tmp_path, monkeypatch):
    op = "zzz_chain_probe"
    op_dir = tmp_path / op
    op_dir.mkdir()
    (op_dir / "patch.py").write_text(
        "NEW_DIM_SYMBOLS = []\n"
        "OP_REGISTRATIONS = '''\n"
        "register_op(OpSpec(\"zzz_chain_probe\",\n"
        "    flops_fn=_zero, params_fn=_zero,\n"
        "    shape_in=(\"B\",), shape_out=(\"B\",),\n"
        "    shape_kind=\"preserve\", layout_family=\"linear\",\n"
        "    formula=\"y = x\", description=\"probe\", required_dims=(),\n"
        "))\n'''\n",
        encoding="utf-8",
    )
    (op_dir / "review.json").write_text(
        json.dumps({"verdict": "auto_merge", "red_flags": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(chain_mod, "PROPOSALS_DIR", str(tmp_path))

    class _Locker:
        def acquire(self, key, *, consume_slot=True):
            return "ok"

        def release(self, key):
            return None

        def stream_agent(self, *a, **k):
            yield from _ok_done()

        def prepare_review_prompt(self, op_name):
            return {"prompt": "x"}

    fake = _Locker()
    monkeypatch.setattr(chain_mod, "op_proposal", fake)
    monkeypatch.setattr(chain_mod, "op_review", fake)

    result = chain_mod.run_propose_review(
        op, {"op_name": op, "context_model_id": "org/M"},
        consume_slot=False,
    )
    assert result.ready is True
    assert result.patch_path == str(op_dir / "patch.py")
    assert any("zzz_chain_probe" in body for body in result.spliced.values())
    # Serving registry file on disk is unchanged — spliced_files is in-memory.
    from backend.arch.promote_op_proposal import OPS_REGISTRY_PATH
    assert "zzz_chain_probe" not in OPS_REGISTRY_PATH.read_text(encoding="utf-8")
