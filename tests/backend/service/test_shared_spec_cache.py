"""backend.service.shared_spec_cache — Phase A cross-VM spec cache.

Two integration paths (write-through + read-through), all Supabase
calls intercepted via pytest-httpx. The cache module silent-fails
on every error; tests pin the contract that no exception ever leaks
out to break /analyze or /generate-spec.
"""
from __future__ import annotations

import re

import pytest


URL = "https://test-spec-cache.supabase.co"
UPSERT_RE = re.compile(re.escape(URL) + r"/rest/v1/rpc/upsert_cached_spec")
GET_RE    = re.compile(re.escape(URL) + r"/rest/v1/rpc/get_cached_spec")
BUMP_RE   = re.compile(re.escape(URL) + r"/rest/v1/rpc/bump_cached_spec_hit")


@pytest.fixture
def supabase_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", URL)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test")


@pytest.fixture
def no_supabase(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)


# ---------------------------------------------------------------------------
# Plumbing: silent-fail when env unset
# ---------------------------------------------------------------------------

class TestSilentFail:
    def test_upsert_skipped_when_unconfigured(self, no_supabase):
        from backend.service import shared_spec_cache
        # No httpx_mock — would explode if a real call escaped.
        ok = shared_spec_cache.upsert_after_success(
            "x/y", spec_yaml="components:\n  attention: gqa_attention\n",
        )
        assert ok is False

    def test_fetch_returns_none_when_unconfigured(self, no_supabase):
        from backend.service import shared_spec_cache
        assert shared_spec_cache.fetch("x/y") is None

    def test_bump_no_op_when_unconfigured(self, no_supabase):
        from backend.service import shared_spec_cache
        # Should not raise.
        shared_spec_cache.bump_hit("x/y")

    def test_supabase_outage_swallowed_on_upsert(self, supabase_env, httpx_mock):
        from backend.service import shared_spec_cache
        httpx_mock.add_response(
            method="POST", url=UPSERT_RE,
            status_code=500, text="db down",
        )
        ok = shared_spec_cache.upsert_after_success(
            "x/y", spec_yaml="dummy",
        )
        assert ok is False  # silent failure surfaces as bool, no raise

    def test_network_error_swallowed_on_fetch(self, supabase_env, monkeypatch):
        """RequestError mid-fetch → returns None, doesn't raise."""
        from backend.service import shared_spec_cache

        def _explode(*a, **kw):
            raise RuntimeError("simulated network failure")

        # Replace the actual http call. Easiest: monkeypatch httpx.Client
        # in the module to throw on POST.
        import httpx

        def _bad_post(self, *a, **kw):
            raise httpx.RequestError("simulated network failure")
        monkeypatch.setattr(httpx.Client, "post", _bad_post)

        assert shared_spec_cache.fetch("x/y") is None


# ---------------------------------------------------------------------------
# Write-through (upsert)
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_upsert_passes_through_payload(self, supabase_env, httpx_mock):
        from backend.service import shared_spec_cache
        captured = {}
        def _capture(req):
            import json
            captured["body"] = json.loads(req.content)
            from httpx import Response
            return Response(200, text="")
        httpx_mock.add_callback(_capture, method="POST", url=UPSERT_RE)

        ok = shared_spec_cache.upsert_after_success(
            "meta-llama/Llama-4",
            spec_yaml="match:\n  model_type: [llama4]\n",
            snapshot_json={"total_params": 70_000_000_000},
            new_component_yamls=[
                {"path": "backend/arch/components/library/llama4_attn.yaml",
                 "body": "name: llama4_attn\n..."},
            ],
            agent_cli="claude",
            generated_by="alice-uuid",
        )
        assert ok is True
        body = captured["body"]
        assert body["p_model_id"] == "meta-llama/Llama-4"
        assert "llama4" in body["p_spec_yaml"]
        assert body["p_snapshot_json"] == {"total_params": 70_000_000_000}
        assert len(body["p_new_component_yamls"]) == 1
        assert body["p_agent_cli"] == "claude"
        assert body["p_generated_by"] == "alice-uuid"

    def test_upsert_optional_fields_default(self, supabase_env, httpx_mock):
        from backend.service import shared_spec_cache
        captured = {}
        def _capture(req):
            import json
            captured["body"] = json.loads(req.content)
            from httpx import Response
            return Response(200, text="")
        httpx_mock.add_callback(_capture, method="POST", url=UPSERT_RE)

        shared_spec_cache.upsert_after_success("x/y", spec_yaml="dummy")
        body = captured["body"]
        assert body["p_snapshot_json"] == {}
        assert body["p_new_component_yamls"] == []
        assert body["p_agent_cli"] is None
        assert body["p_generated_by"] is None


# ---------------------------------------------------------------------------
# Read-through (fetch)
# ---------------------------------------------------------------------------

class TestFetch:
    def test_hit_returns_first_row(self, supabase_env, httpx_mock):
        from backend.service import shared_spec_cache
        row = {
            "model_id": "meta-llama/Llama-4",
            "spec_yaml": "match:\n  model_type: [llama4]\n",
            "snapshot_json": {"k": "v"},
            "new_component_yamls": [],
            "agent_cli": "claude",
            "hit_count": 7,
            "created_at": "2026-05-05T00:00:00Z",
            "last_used_at": "2026-05-05T00:00:00Z",
        }
        httpx_mock.add_response(method="POST", url=GET_RE, json=[row])
        result = shared_spec_cache.fetch("meta-llama/Llama-4")
        assert result == row

    def test_miss_returns_none(self, supabase_env, httpx_mock):
        from backend.service import shared_spec_cache
        httpx_mock.add_response(method="POST", url=GET_RE, json=[])
        assert shared_spec_cache.fetch("not-cached/model") is None

    def test_supabase_500_returns_none(self, supabase_env, httpx_mock):
        from backend.service import shared_spec_cache
        httpx_mock.add_response(
            method="POST", url=GET_RE,
            status_code=500, text="boom",
        )
        assert shared_spec_cache.fetch("any/model") is None


# ---------------------------------------------------------------------------
# Bump hit
# ---------------------------------------------------------------------------

class TestBump:
    def test_bump_calls_correct_rpc(self, supabase_env, httpx_mock):
        from backend.service import shared_spec_cache
        captured = {}
        def _capture(req):
            import json
            captured["body"] = json.loads(req.content)
            from httpx import Response
            return Response(200, text="")
        httpx_mock.add_callback(_capture, method="POST", url=BUMP_RE)

        shared_spec_cache.bump_hit("x/y")
        assert captured["body"] == {"p_model_id": "x/y"}

    def test_bump_silent_on_500(self, supabase_env, httpx_mock):
        from backend.service import shared_spec_cache
        httpx_mock.add_response(
            method="POST", url=BUMP_RE,
            status_code=500, text="boom",
        )
        # Must not raise.
        shared_spec_cache.bump_hit("x/y")


# ---------------------------------------------------------------------------
# Loader integration: SpecLoader.match falls back to cache on miss
# ---------------------------------------------------------------------------

class TestLoaderReadThrough:
    """The cache reaches the loader through the spec-miss hook only. The hook
    lives in the service layer, refuses in multi-user mode, and writes the
    row into a serving overlay under `HA_USER_SPECS_ROOT`; the loader
    attaches that overlay and never sees Supabase or the repo tree."""

    @pytest.fixture(autouse=True)
    def _self_host_hook(self, monkeypatch, tmp_path):
        import backend.arch.loader as loader_mod
        from backend.service import shared_spec_cache
        # auth_enabled() reads SUPABASE_* too, so pin self-host explicitly.
        monkeypatch.setattr("backend.supabase_env.jwt_verification_available", lambda: False)
        monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
        monkeypatch.setattr(loader_mod, "_spec_miss_hook",
                            shared_spec_cache.materialize_cached_spec)
        self.specs_root = tmp_path / "user_specs"

    def test_match_with_unknown_model_id_falls_back_to_cache(self, supabase_env, httpx_mock):
        """A cached row becomes an overlay, the loader attaches it, and the
        config matches; the repo tree is untouched."""
        from pathlib import Path
        from backend.arch.loader import SpecLoader
        import backend.arch.loader as loader_mod
        cached_yaml = (
            "match:\n"
            "  model_type: [synthetic_test_arch]\n"
            "  model_id: [fake/synthetic]\n"
            "dims:\n"
            "  d_model: config.hidden_size\n"
            "  n_vocab: config.vocab_size\n"
            "  n_layers: config.num_hidden_layers\n"
            "blocks:\n"
            "  layer:\n"
            "    graph:\n"
            "      y: { op: linear, in: x, dims: { in: d_model, out: d_model } }\n"
            "    outputs: [y]\n"
            "model:\n"
            "  - { name: embed_tokens, op: embedding, boundary_role: input_embedding, dims: { n_vocab: n_vocab, d_model: d_model } }\n"
            "  - { name: layers, repeat: n_layers, template: layer }\n"
            "  - { name: lm_head, op: linear, boundary_role: output_head, dims: { in: d_model, out: n_vocab } }\n"
        )
        row = {
            "model_id": "fake/synthetic",
            "spec_yaml": cached_yaml,
            "snapshot_json": {"model_type": "synthetic_test_arch"},
            "new_component_yamls": [],
            "agent_cli": None,
            "hit_count": 1,
            "created_at": "2026-05-05T00:00:00Z",
            "last_used_at": "2026-05-05T00:00:00Z",
        }
        httpx_mock.add_response(method="POST", url=GET_RE, json=[row])
        httpx_mock.add_response(method="POST", url=BUMP_RE, text="", is_optional=True)

        loader = SpecLoader()
        config = {
            "model_type": "synthetic_test_arch",
            "hidden_size": 4096,
            "vocab_size": 128,
            "num_hidden_layers": 32,
        }
        result = loader.match(config, model_id="fake/synthetic")
        assert result is not None
        assert result.spec.name == "fake__synthetic"

        overlay = self.specs_root / "_cached_specs" / "fake__synthetic"
        assert (overlay / "models" / "fake__synthetic.yaml").is_file()
        assert (overlay / "snapshots" / "fake__synthetic.json").is_file()
        repo_models = Path(loader_mod.__file__).resolve().parent / "models"
        assert not (repo_models / "fake__synthetic.yaml").exists()

    def test_retired_row_is_a_miss_and_leaves_the_loader_unchanged(self, supabase_env, httpx_mock):
        """A row in a format the loader has since retired is a miss. The
        overlay is written (it is a cache) but nothing from it is registered,
        so the model_type is not claimed by a failed spec and the next call
        is a fresh attempt."""
        from backend.arch.loader import SpecLoader
        retired_yaml = (
            "match:\n"
            "  model_type: [synthetic_test_arch]\n"
            "  model_id: [fake/synthetic]\n"
            "dims:\n"
            "  d_model: config.hidden_size\n"
            "  n_layers: config.num_hidden_layers\n"
            "blocks:\n"
            "  layer:\n"
            "    name: layer\n"
            "    graph:\n"
            "      y: { op: linear, in: x, dims: { in: d_model, out: d_model } }\n"
            "    outputs: [y]\n"
            "model: []\n"
        )
        row = {
            "model_id": "fake/retired",
            "spec_yaml": retired_yaml,
            "snapshot_json": {},
            "new_component_yamls": [],
        }
        httpx_mock.add_response(method="POST", url=GET_RE, json=[row])
        httpx_mock.add_response(method="POST", url=BUMP_RE, text="", is_optional=True)

        loader = SpecLoader()
        before = dict(loader._specs)
        config = {"model_type": "synthetic_test_arch", "hidden_size": 8,
                  "num_hidden_layers": 1}
        assert loader.match(config, model_id="fake/retired") is None
        assert loader._specs == before
        assert loader._model_load_failures == []

        httpx_mock.add_response(method="POST", url=GET_RE, json=[])
        assert loader.match(config, model_id="fake/retired") is None

    def test_multiuser_never_serves_the_global_cache(self, supabase_env, httpx_mock, monkeypatch):
        """With auth on, the hook answers None before touching Supabase."""
        from backend.service import shared_spec_cache
        monkeypatch.setattr("backend.supabase_env.jwt_verification_available", lambda: True)
        # No httpx_mock response registered: a fetch would fail the test.
        assert shared_spec_cache.materialize_cached_spec("fake/synthetic") is None

    def test_the_gate_is_an_environment_read_not_a_backend_call(self, supabase_env, httpx_mock, monkeypatch):
        """B15: the old hook wrapped the switch in `except Exception` — one
        side failing open, the other closed. The switch reads two env vars
        and cannot fail; with the URL set nothing is fetched (no httpx
        response is registered, so a request would fail this test)."""
        from backend.service import shared_spec_cache
        monkeypatch.setattr("backend.supabase_env.jwt_verification_available", lambda: True)
        assert shared_spec_cache.materialize_cached_spec("fake/synthetic") is None

    def test_match_without_model_id_skips_cache(self, supabase_env, httpx_mock):
        """Callers that don't pass model_id never reach the hook."""
        from backend.arch.loader import SpecLoader
        # No httpx_mock.add_response — if a network call escapes, the
        # test fails. Confirms cache is bypassed.
        loader = SpecLoader()
        result = loader.match({"model_type": "synthetic_test_arch"}, model_id=None)
        # Won't match anything (no built-in spec claims synthetic_test_arch),
        # but importantly: didn't crash, didn't hit network.
        assert result is None

    def test_match_without_supabase_skips_cache(self, no_supabase):
        """No SUPABASE env → no cache call even when model_id passed."""
        from backend.arch.loader import SpecLoader
        loader = SpecLoader()
        result = loader.match(
            {"model_type": "synthetic_test_arch"},
            model_id="never/never",
        )
        assert result is None
