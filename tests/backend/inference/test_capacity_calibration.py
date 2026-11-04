"""Capacity identity and ceiling contracts."""
from __future__ import annotations


from tests.snapshots import resolve_snapshot, snapshot_config

def _resolve_llama32_3b():
    """Best-effort resolve of meta-llama/Llama-3.2-3B; skip if the
    snapshot isn't on disk."""

    cfg = snapshot_config("meta-llama/Llama-3.2-3B")
    resolved = resolve_snapshot("meta-llama/Llama-3.2-3B")
    return cfg, resolved


class TestSpecHashContentFingerprint:
    """`ResolvedSpec.identity_hash` must hash the resolved spec's STRUCTURAL
    CONTENT, not just metadata (name + yaml path + dim_values +
    model_type). Otherwise edits to a spec yaml's blocks, a referenced
    component, or a dim formula leave the hash unchanged — existing
    benchmark CSV rows then exact-match against a structurally-
    different graph, defeating the purpose of `spec_hash` in the
    calibration identity tuple."""

    def test_modifying_block_graph_changes_hash(self):
        """If the spec's structural body is edited (a block-graph
        node renamed, a dim formula tweaked, a component reference
        swapped), `ResolvedSpec.identity_hash` must change. Otherwise stale CSV
        measurements would silently match a structurally-different
        graph.

        This test mutates a DEEP structural field — `spec.components`
        (the resolved block-graph dict). A metadata-only hash would
        leave the value identical; a Pydantic `model_dump`-based
        fingerprint walks `spec.components` and the change shows up.

        The mutant is a real `ResolvedSpec` carrying the edited spec: the
        hash is the resolution's own answer (`ResolvedSpec.identity_hash`),
        so a stand-in with the three fields it reads would be testing an
        assembly nothing in the repo performs.
        """
        from copy import deepcopy
        _cfg, resolved = _resolve_llama32_3b()
        baseline = resolved.identity_hash()

        spec_copy = deepcopy(resolved.spec)
        components = getattr(spec_copy, "components", None)
        if not components:
            self.skipTest("spec has no components dict to mutate")

        # Rename one component key — a structural change that does
        # NOT touch any metadata-only hash inputs.
        old_key = next(iter(components))
        components[old_key + "_CONTENT_FINGERPRINT_PROBE"] = components.pop(old_key)

        mut = resolved.with_spec(spec_copy)

        mutated_hash = mut.identity_hash()
        assert baseline != mutated_hash, (
            f"renaming a key in spec.components (deep structural "
            f"content) must change spec_hash. "
            f"baseline={baseline!r} mutated={mutated_hash!r}. "
            f"A metadata-only hash would not detect this."
        )

    def test_modifying_dim_values_changes_hash(self):
        """Same content, different dim binding MUST hash differently —
        lock it in alongside the content fingerprint so a future
        refactor doesn't drop it."""
        _cfg, resolved = _resolve_llama32_3b()
        baseline = resolved.identity_hash()

        mut = resolved.with_spec(resolved.spec)
        dims = dict(mut.dim_values)
        # Bump any dim value to force a different binding.
        first_key = sorted(dims)[0]
        dims[first_key] = dims[first_key] + 1
        mut.dim_values = dims

        assert baseline != mut.identity_hash(), (
            "different dim_values must produce different spec_hash"
        )

    def test_identical_resolved_specs_produce_identical_hash(self):
        """Two resolves of the same snapshot through the same loader
        MUST produce byte-identical hashes (stability / determinism
        property — required for CSV identity bytes to be stable
        across processes / OS / Python versions)."""
        _cfg, resolved_a = _resolve_llama32_3b()
        _cfg2, resolved_b = _resolve_llama32_3b()
        assert resolved_a.identity_hash() == resolved_b.identity_hash()


class TestCapacityCeilingsLayer:
    """L1 (`capacity_ceilings`) is the capacity layer: does it fit, at what
    batch. `estimate_capacity` composes L1..L4 on top. A gate that calls L1
    directly must see exactly the ceilings the full stack reports — if the two
    ever diverge, the sweep would silently sample a different set of cells."""

    def _cfg(self):
        from backend.analyzer import fetch_config
        return fetch_config("Qwen/Qwen3-30B-A3B")

    def _resolved(self, cfg):
        """Resolve once so capacity never re-matches (single entry)."""
        from tests.snapshots import shared_loader
        return shared_loader().match(cfg, model_id="Qwen/Qwen3-30B-A3B")

    def test_l1_matches_the_full_composition(self):
        from backend.inference.capacity import (
            estimate_capacity, estimate_capacity_ceilings,
        )
        from backend.inference.parallel_config import ParallelConfig
        cfg = self._cfg()
        resolved = self._resolved(cfg)
        for seq in (2048, 8192):
            kw = dict(config=cfg, resolved=resolved, gpu_sku="H100-SXM",
                      parallel=ParallelConfig(tp_attention=4, tp_dense_ffn=4),
                      num_gpus=4, avg_input_tokens=seq, avg_output_tokens=256)
            full = estimate_capacity(**kw)
            l1 = estimate_capacity_ceilings(**kw)
            assert l1.max_batch == full["max_decode_batch_per_gpu"]
            assert l1.max_batch == full["max_batch_per_gpu"]
            assert l1.max_prefill_batch == full["max_prefill_batch_per_gpu"]

    def test_a_request_past_the_published_context_is_flagged(self):
        # The model's published context (dim `s_max`) is on every estimate,
        # and a request whose prompt runs past it says so: the formulas do
        # not stop at the limit, so a page must be told. A prompt AT the
        # limit is served (the output is what the stack truncates).
        from backend.inference.capacity import estimate_capacity
        from backend.inference.parallel_config import ParallelConfig
        cfg = self._cfg()
        resolved = self._resolved(cfg)
        limit = resolved.dim_values["s_max"]
        kw = dict(config=cfg, resolved=resolved, gpu_sku="H100-SXM",
                  parallel=ParallelConfig(tp_attention=4, tp_dense_ffn=4),
                  num_gpus=4, avg_output_tokens=256)
        inside = estimate_capacity(avg_input_tokens=limit, **kw)
        past = estimate_capacity(avg_input_tokens=limit + 1, **kw)
        assert inside["max_context_tokens"] == past["max_context_tokens"] == limit
        assert inside["beyond_max_context"] is False
        assert past["beyond_max_context"] is True

    def test_l1_stops_at_l1(self):
        # The layer returns capacity numbers — it does not carry phases,
        # per-op rows or formulas. Those are what the layers above add.
        from backend.inference.capacity import estimate_capacity_ceilings
        cfg = self._cfg()
        l1 = estimate_capacity_ceilings(
            config=cfg, resolved=self._resolved(cfg), gpu_sku="H100-SXM",
            num_gpus=1,
            avg_input_tokens=2048, avg_output_tokens=256,
            )
        assert isinstance(l1.max_batch, int)
        for absent in ("phases", "per_op_timing", "formulas"):
            assert not hasattr(l1, absent)
