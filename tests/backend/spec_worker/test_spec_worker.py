import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from backend import spec_worker
# `hf_metadata` was reorganized into `backend.arch.quant_schemes` —
# alias kept so the existing test cases below don't need rewriting.
from backend.arch import quant_schemes as hf_metadata


class TestSpecWorker(unittest.TestCase):
    def test_scale_inv_classified_as_quant_metadata(self):
        fp8 = hf_metadata.QUANT_SCHEMES["fp8"]
        self.assertTrue(hf_metadata.is_quantization_metadata("model.layers.0.q_proj.weight_scale_inv", fp8))
        self.assertFalse(hf_metadata.is_quantization_metadata("model.layers.0.q_proj.weight", fp8))

    def test_agent_cli_defaults_to_claude(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(spec_worker.get_agent_cli(), "claude")
            self.assertEqual(spec_worker.get_agent_binary(), "claude")
            self.assertEqual(spec_worker.get_agent_display_name(), "Claude Code")

    def test_agent_cli_can_switch_to_codex(self):
        with patch.dict(os.environ, {"SPEC_AGENT_CLI": "codex"}, clear=True):
            self.assertEqual(spec_worker.get_agent_cli(), "codex")
            self.assertEqual(spec_worker.get_agent_binary(), "codex")
            self.assertEqual(spec_worker.get_agent_display_name(), "Codex")

    def test_invalid_agent_cli_raises(self):
        with patch.dict(os.environ, {"SPEC_AGENT_CLI": "invalid"}, clear=True), \
                self.assertRaises(ValueError):
            spec_worker.get_agent_cli()

    def test_configure_runtime_updates_agent(self):
        with patch.dict(os.environ, {}, clear=True):
            spec_worker.configure_runtime(agent_cli="claude", max_concurrent=3)
            self.assertEqual(spec_worker.get_agent_cli(), "claude")

    def test_parse_codex_failure_event(self):
        from backend.agents.codex import CodexBackend
        events = CodexBackend().parse_event({
            "type": "turn.failed",
            "error": {"message": "stream disconnected"},
        })
        self.assertEqual(events, [{
            "type": "done",
            "success": False,
            "message": "stream disconnected",
        }])

    def test_parse_codex_nested_text(self):
        from backend.agents.codex import CodexBackend
        events = CodexBackend().parse_event({
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "validation passed"},
                ],
            },
        })
        self.assertEqual(events, [{"type": "log", "text": "validation passed"}])

    def test_extract_usage_events_from_prompt_completion_fields(self):
        from backend.agents.usage import extract_usage_events
        events = extract_usage_events({
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            }
        })
        self.assertEqual(events, [{
            "type": "usage",
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
        }])

    def test_extract_usage_events_from_nested_input_output_fields(self):
        from backend.agents.usage import extract_usage_events
        events = extract_usage_events({
            "response": {
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 80,
                }
            }
        })
        # total_tokens includes cache_read so cache-hit turns aren't
        # misreported as 49 tokens when the agent really processed 68K.
        self.assertEqual(events, [{
            "type": "usage",
            "input_tokens": 200,
            "output_tokens": 50,
            "total_tokens": 330,
            "cache_read_input_tokens": 80,
        }])

    def test_build_prefetch_logs_for_model_info_failure(self):
        from backend.spec_worker._stream import build_prefetch_logs
        logs = build_prefetch_logs({
            "config": {
                "model_type": "deepseek_v32",
                "transformers_version": "4.44.2",
            },
            "model_info_status": {
                "success": False,
                "transformers_version": "5.2.0",
                "error": "model type `deepseek_v32` not recognized",
            },
        })
        self.assertTrue(any("could not build this model skeleton" in line for line in logs))
        self.assertTrue(any("deepseek_v32" in line for line in logs))

    def test_generate_task_body_evaluates_without_format_error(self):
        """Catches a regression where the task-body f-string contained
        unescaped `{ ... }` from a literal YAML schema example. The
        f-string parser then sees `{ 4: 8, 8: 4 }` as expression `4` +
        format spec ` 8, 8: 4 `, raising `ValueError: Invalid format
        specifier`. The agent's prompt never reaches Anthropic; the
        worker bubbles the error up as `Metadata fetch failed: Invalid
        format specifier`, which is hard to diagnose. Cheapest possible
        guard: just call the builder.
        """
        from backend.spec_worker._prompts.tasks import build_generate_task_body
        out = build_generate_task_body("test/model", {})
        self.assertIn("Generate spec", out)
        self.assertIn("test/model", out)

    def test_system_prompt_template_format_clean(self):
        """Sibling regression to the task-body f-string check, but for
        the markdown template path: `system.py` runs
        `_load_template("system.md").format(ops_doc=..., components=...,
        schema_doc=..., examples=...)`. Any literal `{ ... }` in the
        template that ISN'T one of those four placeholders has to be
        escaped as `{{ ... }}` — otherwise format() raises KeyError on
        a synthetic placeholder name (e.g. `KeyError: ' op'` from a YAML
        snippet `{ op: foo }` left unescaped in a docs example).

        The bug surfaces to the user as `Metadata fetch failed: ' op'`
        with no further hint. Test exercises the exact format() call
        the spec-gen worker makes; cache cleared so the test sees the
        current template content even when run in the same process as
        a prior caller.
        """
        from backend.spec_worker._prompts import system as _sys
        _sys._load_template.cache_clear()
        prompt = _sys.build_system_prompt()
        self.assertIn("HuggingArch Spec System", prompt)

    def test_reuse_priority_names_bases_before_new_drafts(self):
        """The authoring policy is observable in both prompt layers."""
        from backend.spec_worker._prompts import system as _sys
        from backend.spec_worker._prompts.tasks import build_generate_task_body

        _sys._load_template.cache_clear()
        system = _sys.build_system_prompt()
        task = build_generate_task_body("test/model", {})
        for prompt in (system, task):
            self.assertIn("thin identity spec", prompt)
            self.assertIn("merged draft", prompt)
            self.assertIn("new draft", prompt)
        self.assertIn("## Authoring priority", system)
        self.assertIn("**Phase 3 reuse priority**", task)
        self.assertIn("## Formal Base Inventory", system)
        self.assertIn("`qwen2`", system)
        self.assertLess(
            system.index("write a thin identity spec"),
            system.index("Reuse an already merged draft"),
        )
        self.assertLess(
            system.index("Reuse an already merged draft"),
            system.index("author a new draft"),
        )

    def test_system_prompt_matches_authored_schema_contracts(self):
        """Keep the writer's contract synchronized with schema validation."""
        from backend.spec_worker._prompts import system as _sys

        _sys._load_template.cache_clear()
        prompt = _sys.build_system_prompt()
        for contract in (
            "Every model row chooses exactly one structural form",
            "exactly one top-level input boundary",
            "checkpoint address is mandatory",
            "Every direct occurrence of `embedding`, `linear`",
            "`linear`, `linear_param`",
            "`per_head_linear`, `conv1d`, `rmsnorm`",
            "exactly duplicated selector is rejected",
            "`auto_imported` is regeneration metadata",
            "Checkpoint naming is declared with `module` and `inner_modules`",
            "`module:` / `inner_modules:`",
            "Eval binds a reader to the producer tensor",
            "`shape` is not an eval seed",
            "`decode_variant` names the component used at decode",
        ):
            self.assertIn(contract, prompt)
        self.assertNotIn("`rename:`).", prompt)
        self.assertNotIn("+ `rename:`", prompt)


class TestSmartTruncate(unittest.TestCase):
    """Prompt budget: forward() bodies survive whole; __init__ pays."""

    def _file(self, n_init_lines: int, n_fwd_lines: int) -> str:
        init = "\n".join(f"        self.p{i} = {i}" for i in range(n_init_lines))
        fwd = "\n".join(f"        x = x + {i}" for i in range(n_fwd_lines))
        return ("import torch\n\n"
                "class Block(torch.nn.Module):\n"
                "    def __init__(self):\n"
                f"{init}\n\n"
                "    def forward(self, x):\n"
                f"{fwd}\n"
                "        return x\n")

    def test_under_budget_passes_through(self):
        from backend.spec_worker._metadata import _fit_file_to_budget
        body = self._file(5, 5)
        assert _fit_file_to_budget(body, 10_000) == body

    def test_over_budget_keeps_forward_whole(self):
        from backend.spec_worker._metadata import _fit_file_to_budget
        body = self._file(4000, 40)   # huge __init__, small forward
        out = _fit_file_to_budget(body, 4_000)
        assert "def forward(self, x):" in out
        assert "return x" in out                       # forward's TAIL survived
        assert "smart-truncated" in out
        assert "self.p3999" not in out                 # __init__ boilerplate cut

    def test_unparseable_falls_back_to_byte_cut(self):
        from backend.spec_worker._metadata import _fit_file_to_budget
        body = "not python (((" + "x" * 10_000
        out = _fit_file_to_budget(body, 100)
        assert "truncated" in out and "smart" not in out


class TestVerifySpecOutput(unittest.TestCase):
    """The success-verdict gate: a REFINE run (seeded=True) whose seeded spec is
    already valid (the agent correctly made no edit) must be judged on the spec
    that exists, not downgraded to 'nothing generated' — while a GENERATE run
    (seeded=False) that reuses the overlay leaf must NOT pass on a stale yaml.
    `_op_chain.find_fresh_gap` is stubbed to None so the no-fresh-write path
    reaches the seeded-spec check."""

    def _overlay_with_spec(self, tmp, model_id):
        from backend.spec_worker.overlay import _safe
        models = Path(tmp) / "models"
        models.mkdir(parents=True)
        yaml_path = models / f"{_safe(model_id)}.yaml"
        yaml_path.write_text("name: x\n", encoding="utf-8")
        past = os.stat(yaml_path).st_mtime - 1000  # OLD: not written this run
        os.utime(yaml_path, (past, past))
        return yaml_path

    def test_refine_noop_but_valid_seeded_spec_is_success(self):
        import io
        import tempfile
        from backend.spec_worker import worker
        model_id = "org/Model-X"
        with tempfile.TemporaryDirectory() as tmp:
            self._overlay_with_spec(tmp, model_id)
            with patch("backend.spec_worker._op_chain.find_fresh_gap", return_value=None), \
                 patch("backend.spec_worker.worker._invoke_validate",
                       return_value={"valid": True, "errors": []}) as v:
                downgrade, result = worker._verify_spec_output(
                    model_id, io.StringIO(), started_at=1e18, root=tmp, seeded=True)
            self.assertIsNone(downgrade)                 # success, not a downgrade
            self.assertEqual(result, {"valid": True, "errors": []})
            v.assert_called_once()

    def test_refine_noop_still_broken_downgrades_on_real_errors(self):
        import io
        import tempfile
        from backend.spec_worker import worker
        model_id = "org/Model-X"
        with tempfile.TemporaryDirectory() as tmp:
            self._overlay_with_spec(tmp, model_id)
            with patch("backend.spec_worker._op_chain.find_fresh_gap", return_value=None), \
                 patch("backend.spec_worker.worker._invoke_validate",
                       return_value={"valid": False, "errors": ["-30GB unmatched"]}):
                downgrade, result = worker._verify_spec_output(
                    model_id, io.StringIO(), started_at=1e18, root=tmp, seeded=True)
            self.assertIsNotNone(downgrade)
            self.assertEqual(downgrade.kind, "validation")
            self.assertIn("-30GB unmatched", downgrade.reason)
            self.assertIsNone(result)

    def test_empty_generate_overlay_still_reports_no_output(self):
        import io
        import tempfile
        from backend.spec_worker import worker
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "models").mkdir()            # empty: a fresh generate wrote nothing
            with patch("backend.spec_worker._op_chain.find_fresh_gap", return_value=None):
                downgrade, result = worker._verify_spec_output(
                    "org/Model-X", io.StringIO(), started_at=1e18, root=tmp, seeded=False)
            self.assertIsNotNone(downgrade)
            self.assertEqual(downgrade.kind, "no_output")
            self.assertIsNone(result)

    def test_generate_rerun_does_not_pass_on_stale_leaf_yaml(self):
        # grok's BLOCK: `new_run_overlay` reuses the per-(user,model) leaf and
        # keeps a prior run's authored yaml. A failed re-GENERATE (seeded=False,
        # no fresh write) must still report no_output even though a valid yaml
        # sits in the overlay — it must NOT be validated and passed.
        import io
        import tempfile
        from backend.spec_worker import worker
        model_id = "org/Model-X"
        with tempfile.TemporaryDirectory() as tmp:
            self._overlay_with_spec(tmp, model_id)   # stale valid yaml in the leaf
            with patch("backend.spec_worker._op_chain.find_fresh_gap", return_value=None), \
                 patch("backend.spec_worker.worker._invoke_validate",
                       return_value={"valid": True, "errors": []}) as v:
                downgrade, result = worker._verify_spec_output(
                    model_id, io.StringIO(), started_at=1e18, root=tmp, seeded=False)
            self.assertIsNotNone(downgrade)
            self.assertEqual(downgrade.kind, "no_output")
            self.assertIsNone(result)
            v.assert_not_called()                    # never even validated the stale yaml


if __name__ == "__main__":
    unittest.main()


class TestPublishRecordsStampedSpec(unittest.TestCase):
    """The stored generation record must carry `match.model_id`.

    The loader resolves a checkpoint to its spec by model_id, so a record whose
    spec_yaml declares only `model_type` cannot be resolved back out (review /
    analyze / create-pr all read the stored text). `_publish_to_cache` stamps
    the yaml before it writes, and `spec_generations.record` must receive THAT
    text — this pins the ordering so moving the record call above the stamp
    fails here instead of silently writing unresolvable rows again.

    (The stamp transform itself is covered by tests/backend/spec_worker/
    test_self_register.py — this is about who gets the stamped text.)
    """

    def test_record_receives_stamped_spec_yaml(self):
        import io
        import tempfile
        import yaml as _yaml
        from backend.spec_worker import worker
        from backend.spec_worker.overlay import _safe

        model_id = "org/Model-X"
        seen = {}

        def _fake_record(**kw):
            seen.update(kw)
            return {"outcome": "recorded", "id": "r1"}

        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            models.mkdir(parents=True)
            (models / f"{_safe(model_id)}.yaml").write_text(
                "match:\n  model_type: [modelx]\ncomponents:\n  a: b\n",
                encoding="utf-8")
            # Patch the modules themselves, not sys.modules: `_publish_to_cache`
            # resolves its cache client through this already-loaded module, so a
            # sys.modules swap would be ignored and the real Supabase path used.
            from backend.spec_worker import shared_spec_miss as _cache
            from backend.service import spec_generations as _gens
            with patch.object(_gens, "record", _fake_record), \
                 patch.object(_cache, "upsert_after_success",
                              lambda **kw: True), \
                 patch.object(_cache, "materialize_cached_spec",
                              lambda *a, **kw: None), \
                 patch.object(worker.supabase_env, "configured",
                              return_value=True):
                worker._publish_to_cache(
                    model_id, io.StringIO(), started_at=0.0,
                    user_id="user-1", agent_cli="claude", target_root=tmp,
                    val_result={"valid": True, "errors": []}, seeded=True,
                    code_revision="pinned")

        self.assertTrue(seen, "spec_generations.record was never called")
        match = _yaml.safe_load(seen["spec_yaml"])["match"]
        self.assertEqual(match["model_id"], [model_id])
        self.assertEqual(match["model_type"], ["modelx"])


if __name__ == "__main__":
    unittest.main()
