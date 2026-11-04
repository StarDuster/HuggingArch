import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.arch.loader import SpecLoader
from backend.arch.schema import GraphNode


class TestSpecLoader(unittest.TestCase):
    def test_longcat_matches_by_model_id(self):
        # Longcat's config has no model_type (only architectures); it resolves
        # by its registered model_id now, not by alias derivation.
        import json
        snap = PROJECT_ROOT / "backend/arch/snapshots/meituan-longcat__LongCat-Flash-Thinking-2601.json"
        cfg = json.loads(snap.read_text())["config"]
        resolved = SpecLoader().match(
            cfg, model_id="meituan-longcat/LongCat-Flash-Thinking-2601")
        self.assertIsNotNone(resolved)


class TestModelIdMatching(unittest.TestCase):
    """Matching is by model_id: a registered id resolves to its own identity
    spec's architecture; same-model_type variants (Llama-2/Llama-3.2,
    GLM bf16/fp8, Qwen3-Next plain/mtp) each resolve correctly by id, not by a
    model_type tie-break; an unregistered id or no id is a miss."""

    def _cfg(self, stem):
        import json
        return json.loads(
            (PROJECT_ROOT / f"backend/arch/snapshots/{stem}.json").read_text())["config"]

    def test_same_model_type_variants_resolve_by_id(self):
        from backend.arch.loader import arch_fingerprint
        loader = SpecLoader()
        cases = {
            "meta-llama/Llama-3.2-1B": "llama",
            "meta-llama/Llama-2-7b-hf": "llama2",
            "zai-org/GLM-5.2": "glm_moe_dsa",
            "Qwen/Qwen3-Coder-Next": "qwen3_next",
        }
        for mid, base in cases.items():
            stem = mid.replace("/", "__", 1)
            got = loader.match(self._cfg(stem), model_id=mid)
            self.assertIsNotNone(got, mid)
            self.assertEqual(
                arch_fingerprint(got.spec), arch_fingerprint(loader.get(base)), mid)

    def test_unregistered_id_is_a_miss(self):
        loader = SpecLoader()
        self.assertIsNone(
            loader.match(self._cfg("Qwen__Qwen3-8B"), model_id="some/unregistered"))

    def test_no_model_id_is_a_miss(self):
        loader = SpecLoader()
        self.assertIsNone(loader.match(self._cfg("Qwen__Qwen3-8B"), model_id=None))

class TestBlockGraphSourceSchema(unittest.TestCase):
    def test_unknown_component_field_is_rejected(self):
        from backend.arch.spec_parsing import _parse_block_graph

        with self.assertRaisesRegex(ValueError, "Unknown component/block field"):
            _parse_block_graph({
                "graph": {
                    "out": {"op": "rmsnorm", "in": "x", "dim": "d_model"},
                },
                "outputs": ["out"],
                "kv_window": "min(s_kv, s_window)",
            })

    def test_retired_kv_field_names_the_replacement(self):
        from backend.arch.spec_parsing import _parse_block_graph

        with self.assertRaisesRegex(ValueError, "removed KV field"):
            _parse_block_graph({
                "role": "attention",
                "kv_label": "GQA",
                "graph": {
                    "k": {"op": "linear", "in": "x",
                          "dims": {"in": "d_model", "out": "d_model"}},
                },
                "outputs": ["k"],
            })

    def test_name_field_is_rejected(self):
        """A component is named by its file stem. A `name:` line could only
        repeat that or contradict it, and a contradiction would split one
        component into two identities — claims looked up under one name,
        `template:` references resolved under the other."""
        from backend.arch.spec_parsing import _parse_block_graph

        with self.assertRaisesRegex(ValueError, "named by the thing that holds it"):
            _parse_block_graph({
                "name": "plain_gqa",
                "role": "attention",
                "cache": {"family": "GQA", "entries": [{"node": "k", "kind": "kv"}]},
                "graph": {
                    "k": {"op": "linear", "in": "x",
                          "dims": {"in": "d_model", "out": "d_model"}},
                },
                "outputs": ["k"],
            }, identity="plain_gqa")

    def test_derived_fields_are_exactly_the_ones_yaml_cannot_declare(self):
        """`LOADER_DERIVED` has to name the same fields the source-key gate
        refuses. A derived field left unmarked reads as part of the author
        schema; a marked field that IS readable means the loader will happily
        take an authored value for something it also computes, and the two
        can disagree with nothing to notice.
        """
        from backend.arch.spec_parsing import _BLOCK_GRAPH_SOURCE_KEYS
        from backend.arch.schema import BlockGraph

        marked = {
            name for name, f in BlockGraph.model_fields.items()
            if (f.json_schema_extra or {}).get("loader_derived")
        }
        unreadable = set(BlockGraph.model_fields) - set(_BLOCK_GRAPH_SOURCE_KEYS)
        self.assertEqual(
            marked, unreadable,
            f"marked-but-readable: {sorted(marked - unreadable)}; "
            f"unreadable-but-unmarked: {sorted(unreadable - marked)}",
        )

    def test_graph_node_derived_fields_are_not_authorable(self):
        from backend.arch.spec_parsing import (
            _GRAPH_NODE_SOURCE_KEYS, _parse_graph_node,
        )
        from backend.arch.schema import GraphNode

        marked = {
            name for name, field in GraphNode.model_fields.items()
            if (field.json_schema_extra or {}).get("loader_derived")
        }
        self.assertEqual(
            marked, set(GraphNode.model_fields) - set(_GRAPH_NODE_SOURCE_KEYS))
        with self.assertRaisesRegex(ValueError, "loader-derived"):
            _parse_graph_node("x", {
                "op": "rmsnorm", "in": "input", "dim": "d_model",
                "resolved_inputs": [["input", "out"]],
            })

    def test_graph_node_formula_caption_is_display_only(self):
        """`formula:` overrides the card's caption at the call site and
        must not leak into the spec dump — the spec-match fingerprint and
        every golden keyed on it would move because a card got a clearer
        caption. Engine and pricing never read it."""
        n = GraphNode(op="add", inputs=["x"], formula="y = a + b")
        self.assertEqual(n.formula, "y = a + b")
        self.assertNotIn("formula", n.model_dump())

    def test_retired_checkpoint_naming_names_the_replacement(self):
        from backend.arch.spec_parsing import _parse_graph_node

        cases = (
            ("rename", {"qkv": "query_key_value"}, "inner_modules"),
            ("hf_inner", {"qkv": "query_key_value"}, "inner_modules"),
            ("hf", "query_key_value", "module"),
            ("primary", True, "nothing"),
        )
        for field, value, replacement in cases:
            with self.subTest(field=field), \
                    self.assertRaisesRegex(ValueError, replacement):
                _parse_graph_node("attn", {
                    "op": "gqa_fused_qkv_attention",
                    "in": "x",
                    field: value,
                })

    def test_attention_variant_falls_back_to_cache_family(self):
        from backend.arch.spec_parsing import _parse_block_graph

        graph = _parse_block_graph({
            "role": "attention",
            "cache": {"family": "GQA", "entries": [{"node": "k", "kind": "kv"}]},
            "graph": {
                "k": {
                    "op": "linear",
                    "in": "x",
                    "dims": {"in": "d_model", "out": "d_model"},
                },
            },
            "outputs": ["k"],
        })

        self.assertIsNone(graph.attn_variant)
        self.assertEqual(graph.attn_display, "GQA")


class TestBlockGraphStreams(unittest.TestCase):
    """Cross-iteration state as a STREAM: a component declares only its
    interface (`reads` / `stream_out`); the stack declares the stream."""

    def test_stream_read_referenceable_as_input(self):
        """A graph node can reference a stream this component `reads`."""
        from backend.arch.spec_parsing import _parse_block_graph
        bg = _parse_block_graph({
            "reads": ["aux"],
            "graph": {
                "norm": {"op": "rmsnorm", "in": "x", "dim": "d_model"},
                # `aux` is the stream name — should validate without
                # being declared as a regular input or graph node.
                "fused": {"op": "add", "in": ["norm", "aux"], "dim": "d_model"},
                "next_aux": {"op": "linear", "in": "fused", "dims": {"in": "d_model", "out": "d_model"}},
            },
            "outputs": ["fused"],
            "stream_out": {"aux": "next_aux"},
        })
        self.assertEqual(bg.reads, ["aux"])
        self.assertEqual(bg.stream_out, {"aux": "next_aux"})

    def test_stream_producer_target_must_exist(self):
        """stream_out pointing at a non-existent node must error."""
        from backend.arch.spec_parsing import _parse_block_graph
        with self.assertRaisesRegex(Exception, "not found in graph"):
            _parse_block_graph({
                "graph": {
                    "norm": {"op": "rmsnorm", "in": "x", "dim": "d_model"},
                },
                "outputs": ["norm"],
                "stream_out": {"aux": "synthetic_node"},
            })

    def test_stream_read_must_be_consumed(self):
        """A `reads:` no node's `in:` names is a dataflow edge the graph
        claims and does not draw (GLM-5.2's shared DSA layers declared
        `reads: [dsa_topk]` while their score read only Q and K)."""
        from backend.arch.spec_parsing import _parse_block_graph
        with self.assertRaisesRegex(Exception, r"reads:` \['aux'\] but no node"):
            _parse_block_graph({
                "reads": ["aux"],
                "graph": {
                    "norm": {"op": "rmsnorm", "in": "x", "dim": "d_model"},
                },
                "outputs": ["norm"],
            })

    def test_retired_carry_syntax_rejected(self):
        """The removed block-local carry keys must fail loudly, not be
        dropped on the floor — the loader reads keys explicitly, so a
        silent drop would delete the stream a spec meant to declare."""
        from backend.arch.spec_parsing import _parse_block_graph
        for key, value in (("carries", {"aux": "d_model"}),
                           ("carry_outputs", {"aux": "norm"}),
                           ("carry_src", {"aux": "forward line"}),
                           ("carry_init", {"aux": {"kind": "zero"}})):
            with self.assertRaisesRegex(Exception, "stream"):
                _parse_block_graph({
                    key: value,
                    "graph": {
                        "norm": {"op": "rmsnorm", "in": "x", "dim": "d_model"},
                    },
                    "outputs": ["norm"],
                })


class TestMisspelledTemplateIsRefused(unittest.TestCase):
    """A template name that is no component, inline block, or op is refused
    when the spec loads, with the nearest names — wherever it is written."""

    SRC = PROJECT_ROOT / "backend" / "arch" / "models" / "gpt_oss.yaml"

    def _load(self, old, new):
        import tempfile
        from backend.arch.loader import SpecLoader
        text = self.SRC.read_text()
        self.assertIn(old, text)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gpt_oss.yaml"
            path.write_text(text.replace(old, new))
            SpecLoader().load_model_spec(str(path))

    def test_a_misspelled_slot_component(self):
        with self.assertRaisesRegex(ValueError, "components.attention_swa.*"
                                    "swa_bias_sink_attention"):
            self._load("  attention_swa: swa_bias_sink_attention",
                       "  attention_swa: swa_bias_sink_attentoin")

    def test_a_misspelled_block_pattern(self):
        with self.assertRaisesRegex(ValueError, "'pre_nrom'.*'pre_norm'"):
            self._load("  block: pre_norm", "  block: pre_nrom")

    def test_a_misspelled_bound_component(self):
        with self.assertRaisesRegex(ValueError, "param 'attn'.*"
                                    "swa_bias_sink_attention"):
            self._load("swa:  { attention: swa_bias_sink_attention",
                       "swa:  { attention: swa_bias_sink_attentoin")


class TestWeightDimGeometryGuard(unittest.TestCase):
    """A `parameter_vector` must
    not encode storage-packing factors (`/2`, `/8`, …) into its `dim`
    expression. Linear `y = W @ x` has W of shape [Out, In] with
    EXACTLY Out*In elements by matrix-multiplication geometry — there
    is no fractional weight count. A trip here forces re-review of
    the matching quant-scheme yaml under
    backend/arch/quant_schemes/.
    """

    def test_packed_fp4_weight_dim_rejected(self):
        """`I * H / 2` on a `.weight` parameter_vector must fail."""
        from backend.arch.spec_parsing import _parse_block_graph
        with self.assertRaisesRegex(Exception, "literal integer divisor"):
            _parse_block_graph({
                "graph": {
                    "w1.weight": {"op": "parameter_vector", "dim": "I * H / 2"},
                    "w1": {"op": "linear", "in": "x", "dims": {"in": "d_model", "out": "d_ffn"}},
                },
                "outputs": ["w1"],
            })

    def test_int4_packed_weight_dim_rejected(self):
        """Same rule fires on `/8` (INT4 packed in I32)."""
        from backend.arch.spec_parsing import _parse_block_graph
        with self.assertRaisesRegex(Exception, "literal integer divisor"):
            _parse_block_graph({
                "graph": {
                    "qweight": {"op": "parameter_vector", "dim": "I * H / 8"},
                    "out": {"op": "linear", "in": "x", "dims": {"in": "d_model", "out": "d_ffn"}},
                },
                "outputs": ["out"],
            })

    def test_quant_metadata_suffix_pv_rejected(self):
        """Hand-coded `.scale` parameter_vectors are rejected — this
        metadata must come from the engine via
        `quant_context.per_role` + the matching scheme yaml's
        `metadata_siblings:` block, not be authored inline."""
        from backend.arch.spec_parsing import _parse_block_graph
        with self.assertRaisesRegex(Exception, "quantization-metadata suffix"):
            _parse_block_graph({
                "graph": {
                    "w1.weight": {"op": "parameter_vector", "dim": "I * H"},
                    "w1.scale":  {"op": "parameter_vector", "dim": "I * H / 32"},
                    "w1": {"op": "linear", "in": "x", "dims": {"in": "d_model", "out": "d_ffn"}},
                },
                "outputs": ["w1"],
            })

    def test_logical_weight_dim_allowed(self):
        """Pure product of dim symbols on a weight tensor passes."""
        from backend.arch.spec_parsing import _parse_block_graph
        bg = _parse_block_graph({
            "graph": {
                "w.weight": {"op": "parameter_vector", "dim": "I * H"},
                "w": {"op": "linear", "in": "x", "dims": {"in": "d_model", "out": "d_ffn"}},
            },
            "outputs": ["w"],
        })
        self.assertEqual(bg.graph["w.weight"].dim, "I * H")

    def test_quant_suffix_parameter_vector_rejected_at_load(self):
        """parameter_vector named with a quant-metadata suffix is
        rejected at load time. Use `quant_context.per_role` + the
        matching scheme yaml's `metadata_siblings:` declaration."""
        from backend.arch.spec_parsing import _parse_block_graph
        with self.assertRaisesRegex(Exception, "quantization-metadata suffix"):
            _parse_block_graph({
                "graph": {
                    "w1.weight": {"op": "parameter_vector", "dim": "I * H"},
                    "w1.scale":  {"op": "parameter_vector", "dim": "(I / 128) * (H / 128)"},
                    "w1": {"op": "linear", "in": "x", "dims": {"in": "d_model", "out": "d_ffn"}},
                },
                "outputs": ["w1"],
            })

    def test_hc_scale_allowed_as_architectural(self):
        """`hc_*_scale` (hyper-connections per-component scaling) is
        architectural, not quant; must NOT trip the quant-suffix
        lockdown."""
        from backend.arch.spec_parsing import _parse_block_graph
        bg = _parse_block_graph({
            "graph": {
                "hc_attn_scale": {"op": "parameter_vector", "dim": "1"},
                "hc_ffn_base":   {"op": "parameter_vector", "dim": "n_hidden_copies"},
                "norm": {"op": "rmsnorm", "in": "x", "dim": "d_model"},
            },
            "outputs": ["norm"],
        })
        self.assertEqual(bg.graph["hc_attn_scale"].dim, "1")

    def test_norm_and_bias_allowed(self):
        """`.bias` / `_norm` are architectural tensors, must NOT trip
        the quant-suffix lockdown."""
        from backend.arch.spec_parsing import _parse_block_graph
        # No exception expected — these are architectural names.
        _parse_block_graph({
            "graph": {
                "qkv.bias":  {"op": "parameter_vector", "dim": "3 * H"},
                "input_norm.weight": {"op": "parameter_vector", "dim": "d_model"},
                "norm": {"op": "rmsnorm", "in": "x", "dim": "d_model"},
            },
            "outputs": ["norm"],
        })

    def test_a_named_divisor_is_allowed(self):
        """A group or block size divides a tensor's count legitimately once
        it is a declared dim: the divisor then has a source."""
        from backend.arch.spec_parsing import _parse_block_graph
        bg = _parse_block_graph({
            "graph": {
                "hc_attn_base":  {"op": "parameter_vector", "dim": "Hc / n_groups"},
                "norm": {"op": "rmsnorm", "in": "x", "dim": "d_model"},
            },
            "outputs": ["norm"],
        })
        self.assertEqual(bg.graph["hc_attn_base"].dim, "Hc / n_groups")

    def test_a_literal_divisor_is_refused_whatever_the_name(self):
        """A bias or a running statistic counts In × Out-style elements too;
        a literal divisor in its dim has no source, so the name does not
        exempt it."""
        from backend.arch.spec_parsing import _parse_block_graph
        with self.assertRaisesRegex(Exception, "named dim"):
            _parse_block_graph({
                "graph": {
                    "hc_attn_base": {"op": "parameter_vector", "dim": "Hc / 2"},
                    "norm": {"op": "rmsnorm", "in": "x", "dim": "d_model"},
                },
                "outputs": ["norm"],
            })


class TestQuantContext(unittest.TestCase):
    """`quant_context` schema field — load-time validation:
    method ∈ registered quant_schemes, dtype ∈ allowed set,
    overrides shape (exactly one of tensors / tensors_re).
    """

    def _minimal_archspec_kwargs(self):
        """Build a minimal ArchSpec init kwargs dict so each test only
        needs to attach its own quant_context."""
        from backend.arch.schema import (
            BlockGraph, GraphNode, ModelLayer,
        )
        bg = BlockGraph(
            inputs=["x"],
            graph={"norm": GraphNode(op="rmsnorm", inputs=["x"], dim="d_model")},
            outputs=["norm"],
        )
        return dict(
            name="test_spec",
            dims={"d_model": "config.hidden_size"},
            model_inputs={"input_ids": {"kind": "tokens"}},
            blocks={"block": bg},
            model=[
                ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                           boundary_role="input_embedding",
                           dims={"n_vocab": "d_model", "d_model": "d_model"}),
                ModelLayer(name="blk", template="block", repeat="1",
                           inputs=["embed"]),
            ],
        )

    def test_storage_agreement_treats_none_as_unquantized(self):
        from backend.arch.schema import QuantContext
        none = QuantContext(per_role={"default": {"method": "none"}})
        fp8 = QuantContext(per_role={
            "attention": {"method": "fp8", "block_h": 128, "block_w": 128},
        })
        ue8 = QuantContext(per_role={
            "attention": {"method": "fp8_ue8m0", "block_h": 128, "block_w": 128},
        })
        self.assertEqual(none.storage_agreement(""), 1)
        self.assertEqual(none.storage_agreement("fp8"), -1)
        self.assertEqual(fp8.storage_agreement(""), -1)
        self.assertEqual(fp8.storage_agreement("fp8"), 1)
        self.assertEqual(ue8.storage_agreement("fp8"), 1)
        self.assertEqual(fp8.storage_agreement("awq"), -1)

    def test_empty_quant_context_accepted(self):
        from backend.arch.schema import ArchSpec, QuantContext
        spec = ArchSpec(**self._minimal_archspec_kwargs(),
                        quant_context=QuantContext())
        self.assertEqual(spec.quant_context.per_role, {})
        self.assertEqual(spec.quant_context.overrides, [])
        self.assertEqual(spec.quant_context.default_dtype, "BF16")

    def test_quant_context_omitted_is_none(self):
        from backend.arch.schema import ArchSpec
        spec = ArchSpec(**self._minimal_archspec_kwargs())
        self.assertIsNone(spec.quant_context)

    def test_per_role_with_registered_method_accepted(self):
        """fp8 is registered under backend/arch/quant_schemes/."""
        from backend.arch.schema import ArchSpec, QuantContext
        qc = QuantContext(per_role={
            "attention": {"method": "fp8", "block_h": 128, "block_w": 128},
            "default":   {"method": "none"},
        })
        spec = ArchSpec(**self._minimal_archspec_kwargs(), quant_context=qc)
        self.assertEqual(spec.quant_context.per_role["attention"].method, "fp8")
        # Extra fields preserved via Pydantic extra="allow"
        self.assertEqual(
            spec.quant_context.per_role["attention"].model_dump()["block_h"], 128,
        )

    def test_unregistered_method_rejected(self):
        from backend.arch.schema import QuantContext
        with self.assertRaisesRegex(Exception, "not a registered quant scheme"):
            QuantContext(per_role={
                "attention": {"method": "totally-fake-method"},
            })

    def test_per_role_dtype_without_method(self):
        from backend.arch.schema import QuantContext
        qc = QuantContext(per_role={
            "attention": {"dtype": "FP8"},
        })
        self.assertIsNone(qc.per_role["attention"].method)
        self.assertEqual(qc.per_role["attention"].dtype, "FP8")

    def test_none_method_does_not_require_registry(self):
        """`method: none` is the explicit unquantized sentinel; bypasses
        the scheme registry check."""
        from backend.arch.schema import ArchSpec, QuantContext
        qc = QuantContext(per_role={
            "default":      {"method": "none"},
            "ffn_dense":    {"method": "none"},
        })
        spec = ArchSpec(**self._minimal_archspec_kwargs(), quant_context=qc)
        self.assertEqual(spec.quant_context.per_role["default"].method, "none")

    def test_override_with_literal_tensors(self):
        from backend.arch.schema import QuantContext
        qc = QuantContext(overrides=[
            {"tensors": ["embed_tokens.weight", "lm_head.weight"], "dtype": "BF16"},
        ])
        self.assertEqual(qc.overrides[0].tensors,
                         ["embed_tokens.weight", "lm_head.weight"])
        self.assertEqual(qc.overrides[0].dtype, "BF16")

    def test_override_with_regex(self):
        from backend.arch.schema import QuantContext
        qc = QuantContext(overrides=[
            {"tensors_re": r".*self_attn.*", "dtype": "BF16",
             "auto_imported": True},
        ])
        self.assertEqual(qc.overrides[0].tensors_re, r".*self_attn.*")
        self.assertTrue(qc.overrides[0].auto_imported)

    def test_override_must_specify_exactly_one_of_tensors_forms(self):
        """Both tensors + tensors_re → reject. Neither → reject."""
        from backend.arch.schema import QuantContext
        with self.assertRaisesRegex(Exception, "exactly one"):
            QuantContext(overrides=[
                {"tensors": ["x"], "tensors_re": ".*", "dtype": "BF16"},
            ])
        with self.assertRaisesRegex(Exception, "exactly one"):
            QuantContext(overrides=[{"dtype": "BF16"}])

    def test_override_empty_tensor_list_rejected(self):
        from backend.arch.schema import QuantContext
        with self.assertRaisesRegex(Exception, "non-empty"):
            QuantContext(overrides=[{"tensors": [], "dtype": "BF16"}])

    def test_override_unknown_dtype_rejected(self):
        from backend.arch.schema import QuantContext
        with self.assertRaisesRegex(Exception, "not a recognized.*dtype"):
            QuantContext(overrides=[{"tensors": ["x"], "dtype": "BFLOAT16"}])

    def test_default_dtype_unknown_rejected(self):
        from backend.arch.schema import QuantContext
        with self.assertRaisesRegex(Exception, "not a recognized.*dtype"):
            QuantContext(default_dtype="INVALID")

    def test_override_method_fields_are_scheme_checked_and_preserved(self):
        from backend.arch.schema import QuantContext
        qc = QuantContext(overrides=[
            {"tensors_re": r".*kv_scale.*", "dtype": "F8_E4M3",
             "method": "fp8", "block_h": 128, "block_w": 128},
        ])
        self.assertEqual(qc.overrides[0].method, "fp8")
        self.assertEqual(qc.overrides[0].model_extra["block_h"], 128)

    def test_full_dsv4_pro_style_quant_context(self):
        """Worked example from the proposal — DSv4-Pro mixed FP4+FP8."""
        from backend.arch.schema import ArchSpec, QuantContext
        # Use registered role names and fp8 across the board; occurrence-level
        # exceptions belong in overrides rather than invented role names.
        qc = QuantContext(
            per_role={
                "attention":      {"method": "fp8", "block_h": 128, "block_w": 128},
                "ffn_moe":       {"method": "fp8", "block_h": 128, "block_w": 128},
                "ffn_dense":     {"method": "fp8", "block_h": 128, "block_w": 128},
                "default":        {"method": "none"},
            },
            overrides=[
                {"tensors": ["embed.weight"],   "dtype": "BF16"},
                {"tensors_re": r".*_norm\.weight$", "dtype": "BF16",
                 "auto_imported": True},
            ],
            default_dtype="BF16",
        )
        spec = ArchSpec(**self._minimal_archspec_kwargs(), quant_context=qc)
        self.assertEqual(sorted(spec.quant_context.known_methods()), ["fp8"])
        self.assertEqual(len(spec.quant_context.overrides), 2)


class TestInlineBlockParams(unittest.TestCase):
    """Spec yaml may declare a parameterized block in `blocks:` and
    invoke it via `template + params` from `model:`. Before the loader
    fix, `_maybe_synth` early-returned on `template in inline_blocks`,
    so `<param>` placeholders survived into engine eval and raised
    `Node 'attn': unbound op param '<attn>'` at compute time — silent
    at load. This is the natural authoring pattern for model-specific
    decoder blocks (DSv4 hyper-connections), so the loader has to
    support it; these tests pin down that synthesis actually happens
    and the placeholders get substituted."""

    def _write_spec(self, base_dir: Path, yaml_text: str,
                    stem: str = "test_inline") -> SpecLoader:
        # SpecLoader expects `<base>/components/{library,adapters,drafts}`
        # and `<base>/models/`. Empty subdirs are fine for this test —
        # rmsnorm is a registered primitive op, no library lookup needed.
        (base_dir / "components" / "library").mkdir(parents=True)
        (base_dir / "components" / "adapters").mkdir()
        (base_dir / "components" / "drafts").mkdir()
        (base_dir / "models").mkdir()
        # The file stem IS the spec name — a model document has no `name:`
        # field to disagree with it.
        (base_dir / "models" / f"{stem}.yaml").write_text(yaml_text)
        return SpecLoader(base_dir=str(base_dir))

    def test_inline_block_with_params_synthesizes(self):
        import tempfile
        import textwrap
        yaml_text = textwrap.dedent("""\
            match: { model_type: [test_inline_param] }
            dims:
              H: config.hidden_size
            blocks:
              parameterized:
                params:
                  norm: { kind: component }
                graph:
                  n: { op: <norm>, in: x, dim: H }
                outputs: [n]
            model_inputs:
              input_ids: { kind: tokens }
            model:
              - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
                  dims: { n_vocab: H, d_model: H } }
              - { name: layers, repeat: "1", streams: { hidden: { carries: true } }, template: parameterized,
                  params: { norm: rmsnorm }, inputs: [embed] }
              - { name: head, op: linear, boundary_role: output_head,
                  dims: { in: H, out: H }, inputs: [layers] }
        """)
        with tempfile.TemporaryDirectory() as d:
            loader = self._write_spec(Path(d), yaml_text)
            self.assertIn("test_inline", loader._specs)
            spec = loader._specs["test_inline"]
            layer = next(l for l in spec.model if l.name == "layers")
            # The model layer's `template:` must have been rewritten to a
            # synthesized name, not the bare inline block name.
            self.assertNotEqual(
                layer.template, "parameterized",
                "model layer still points at unsynthesized inline block",
            )
            self.assertTrue(
                layer.template.startswith("parameterized__"),
                f"unexpected synthesized name: {layer.template!r}",
            )
            # The synthesized block must have `<norm>` resolved to `rmsnorm`.
            synth = spec.blocks[layer.template]
            for nname, gn in synth.graph.items():
                self.assertFalse(
                    gn.op.startswith("<") and gn.op.endswith(">"),
                    f"node {nname!r} in synthesized block still has "
                    f"unresolved placeholder op={gn.op!r}",
                )

    def test_bare_inline_template_skips_synthesis(self):
        """An inline block referenced WITHOUT params (just `template: foo`)
        should resolve via `ArchSpec.blocks[foo]` directly — synthesizing
        a wrapper around it would just create a self-referential name."""
        import tempfile
        import textwrap
        yaml_text = textwrap.dedent("""\
            match: { model_type: [test_inline_bare] }
            dims:
              H: config.hidden_size
            blocks:
              fixed:
                graph:
                  n: { op: rmsnorm, in: x, dim: H }
                outputs: [n]
            model_inputs:
              input_ids: { kind: tokens }
            model:
              - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
                  dims: { n_vocab: H, d_model: H } }
              - { name: layers, repeat: "1", streams: { hidden: { carries: true } }, template: fixed, inputs: [embed] }
              - { name: head, op: linear, boundary_role: output_head,
                  dims: { in: H, out: H }, inputs: [layers] }
        """)
        with tempfile.TemporaryDirectory() as d:
            loader = self._write_spec(Path(d), yaml_text,
                                      stem="test_inline_bare")
            spec = loader._specs["test_inline_bare"]
            layer = next(l for l in spec.model if l.name == "layers")
            self.assertEqual(layer.template, "fixed")


class TestComponentArityCheck(unittest.TestCase):
    """Loader-time strict arity check on component (BlockGraph)
    invocations. Catches the silent `zip`-drop in engine eval where an
    over-wired caller `in: [a, b]` against a 1-input callee binds
    `x → a` and discards `b`."""

    def _spec_with_component_call(self, caller_inputs):
        """Build a minimal ArchSpec where the top-level block invokes
        a 1-input component `my_ffn`, passing `caller_inputs` as
        the call site's `in:` list."""
        from backend.arch.schema import (
            ArchSpec, BlockGraph, GraphNode, ModelLayer,
        )
        # 1-input component.
        my_ffn = BlockGraph(
            inputs=["x"],
            graph={
                "lin": GraphNode(
                    op="linear", inputs=["x"], dims={"in": "d_model", "out": "d_model"},
                ),
            },
            outputs=["lin"],
        )
        # Top-level block calls `my_ffn` with N inputs (caller-supplied).
        # Need a sibling node so multi-input cases have a real ref to
        # point at without bogus dangling names.
        top = BlockGraph(
            inputs=["x"],
            graph={
                "norm": GraphNode(op="rmsnorm", inputs=["x"], dim="d_model"),
                "ffn":  GraphNode(op="my_ffn", inputs=caller_inputs),
            },
            outputs=["ffn"],
        )
        return ArchSpec(
        model_inputs={"input_ids": {"kind": "tokens"}},
            name="test_arity",
            dims={"d_model": "config.hidden_size"},
            components={"my_ffn": my_ffn},
            blocks={"block": top},
            model=[
                ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                           boundary_role="input_embedding",
                           dims={"n_vocab": "d_model", "d_model": "d_model"}),
                ModelLayer(name="blk", template="block", repeat="1",
                           inputs=["embed"]),
            ],
        )

    def test_matching_arity_accepted(self):
        """1 caller input ↔ 1 callee slot validates cleanly."""
        spec = self._spec_with_component_call(["norm"])
        self.assertEqual(len(spec.blocks["block"].graph["ffn"].inputs), 1)

    def test_over_wired_arity_rejected(self):
        """Calling 1-input component with 2 refs raises ValueError —
        engine's `zip` would silently drop the second ref."""
        with self.assertRaisesRegex(Exception, r"calls component .* with 2 input"):
            self._spec_with_component_call(["norm", "x"])

    def test_under_wired_arity_rejected(self):
        """Calling 1-input component with 0 refs also raises."""
        with self.assertRaisesRegex(Exception, r"calls component .* with 0 input"):
            self._spec_with_component_call([])


class TestTemplateAssignmentsCollision(unittest.TestCase):
    """One layer, one block-binding path. A layer declaring both
    `template:` and `assignments:` lets two layer-0 consumers pick
    differently (spec_tree takes template first) and silently skip or
    double-count the block — and `per_layer:` lowers to `assignments`,
    so the collision is reachable from the authored surface too."""

    def _kwargs(self, with_template: bool, **layer_over):
        from backend.arch.schema import (
            ArchSpec, BlockAssignment, BlockGraph, GraphNode, ModelLayer,
        )
        bg = BlockGraph(
            inputs=["x"],
            graph={"norm": GraphNode(op="rmsnorm", inputs=["x"], dim="d_model")},
            outputs=["norm"],
        )
        layer_kw = {"name": "blk", "repeat": "2", "inputs": ["embed"]}
        if with_template:
            layer_kw["template"] = "block"
        layer_kw.update(layer_over)
        return ArchSpec, dict(
            name="test_spec",
            dims={"d_model": "config.hidden_size"},
            model_inputs={"input_ids": {"kind": "tokens"}},
            blocks={"block": bg},
            model=[
                ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                           boundary_role="input_embedding",
                           dims={"n_vocab": "d_model", "d_model": "d_model"}),
                ModelLayer(**layer_kw),
            ],
        ), BlockAssignment

    def test_template_and_assignments_on_one_layer_rejected(self):
        from backend.arch.schema import BlockAssignment
        with self.assertRaisesRegex(ValueError, "requires exactly one"):
            self._kwargs(
                True,
                assignments=[BlockAssignment(template="block", range=["0", "1"])],
            )

    def test_assignments_alone_accepted(self):
        from backend.arch.schema import BlockAssignment
        ArchSpec, kwargs, _ = self._kwargs(
            False,
            assignments=[BlockAssignment(template="block", range=["0", "1"])],
        )
        spec = ArchSpec(**kwargs)
        self.assertEqual(len(spec.model), 2)


if __name__ == "__main__":
    unittest.main()


class TestEdgeKinds(unittest.TestCase):
    """7.1: edges carry a semantic kind; the boundary validates it."""

    def test_a_mask_edge_to_a_mask_slot_passes(self):
        from backend.arch.schema import BlockGraph, GraphNode
        bg = BlockGraph(
            inputs=["x", "scores"],
            graph={
                "topk_idx": GraphNode(
                    op="dsa_topk_mask", inputs=["scores"],
                    dims={"n_experts_active": "4"}),
                "q": GraphNode(op="view_split", inputs=["x"], axis=-1,
                               dims={"n_heads": "2"}),
                "k": GraphNode(op="view_split", inputs=["x"], axis=-1,
                               dims={"n_heads": "2"}),
                "score": GraphNode(
                    op="dsa_attention_score", inputs=["q", "k", "topk_idx"],
                    dims={"d_head": "64", "n_heads": "2",
                          "n_experts_active": "4"}),
            },
            outputs=["score"],
        )
        self.assertEqual(bg.graph["score"].op, "dsa_attention_score")

    def test_a_mask_edge_to_an_activation_slot_is_rejected(self):
        from backend.arch.schema import BlockGraph, GraphNode
        with self.assertRaisesRegex(ValueError, "kind mismatch"):
            BlockGraph(
                inputs=["scores", "x"],
                graph={
                    "topk_idx": GraphNode(
                        op="dsa_topk_mask", inputs=["scores"],
                        dims={"n_experts_active": "4"}),
                    "n": GraphNode(op="rmsnorm", inputs=["topk_idx"], dim="64"),
                },
                outputs=["n"],
            )

    def test_an_unknown_kind_at_registration_is_rejected(self):
        from backend.arch.ops import OpSpec
        with self.assertRaisesRegex(ValueError, "unknown edge kind"):
            OpSpec(name="bad_op", flops_fn=lambda d: 0, params_fn=lambda d: 0,
                   output_kind="sparkle")

    def test_every_op_kind_is_in_the_vocabulary(self):
        from backend.arch.ops import (EDGE_KIND_VOCABULARY, OP_REGISTRY)
        for name, spec in OP_REGISTRY.items():
            self.assertIn(spec.output_kind, EDGE_KIND_VOCABULARY, name)
            for k in spec.input_kinds:
                self.assertIn(k, EDGE_KIND_VOCABULARY, name)


class TestDerivedFormula(unittest.TestCase):
    """7.4: the compound formula derived from the graph itself."""

    def test_hc_pre_derives_the_sinkhorn_composition(self):
        from backend.arch.loader import SpecLoader
        spec = SpecLoader().all_specs()["deepseek_v4"]
        f = spec.components["hc_pre"].derived_formula()
        assert "hc_split_sinkhorn(" in f
        assert "hc_weighted_reduce(x, split)" in f
        assert "split.post" in f and "split.comb" in f
        # shared nodes are named once, not inlined twice
        assert f.count("view_merge(") == 1


class TestABrokenSpecRefusesByEitherName(unittest.TestCase):
    """A spec that failed to load must refuse BOTH ways of asking for it.

    The failure list has always been consulted on the `match()` path, where
    a broken spec claiming the model_type raises with the parse error. The
    by-name path — what `validate --spec-name` walks, and what spec-gen
    walks the moment it finishes writing a file — went through `get()`,
    got None, and reported "No spec matched": a message about matching for
    a file that never got as far as being matchable. The pydantic error
    naming the exact bad field was already in hand and thrown away.
    """

    RETIRED = (
        # A top-level `name:` — a model spec is named by its file stem, so
        # the schema forbids it. This is what spec-gen wrote for GLM-5.3
        # after copying the GLM-5.1 family spec into a new filename.
        "name: some_other_family\n"
        "match:\n"
        "  model_type: [synthetic_broken_arch]\n"
        "  model_id: [synthetic/broken]\n"
        "dims:\n"
        "  d_model: config.hidden_size\n"
        "model: []\n"
    )

    def _loader(self, tmpdir):
        from backend.arch.loader import SpecLoader, overlay_kwargs
        models = Path(tmpdir) / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "synthetic_broken.yaml").write_text(self.RETIRED)
        return SpecLoader(**overlay_kwargs(tmpdir))

    def test_asking_by_name_raises_the_parse_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            loader = self._loader(tmp)
            with self.assertRaisesRegex(RuntimeError, "failed to load") as ctx:
                loader.resolve_by_name("synthetic_broken", {})
            self.assertIn("name", str(ctx.exception))

    def test_asking_by_model_id_raises_the_parse_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            loader = self._loader(tmp)
            # Matching is by model_id now: a broken spec declaring this id is
            # refused loudly, not a silent miss to the family (contract D).
            with self.assertRaisesRegex(RuntimeError, "failed to load"):
                loader.match({}, model_id="synthetic/broken")

    def test_an_unrelated_name_is_still_an_ordinary_miss(self):
        """The refusal is scoped to the file that broke — a genuinely
        absent spec must stay None, or every miss becomes an exception."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            loader = self._loader(tmp)
            self.assertIsNone(loader.resolve_by_name("no_such_spec", {}))
