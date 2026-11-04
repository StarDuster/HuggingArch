import unittest
import tempfile
import textwrap
from pathlib import Path

from pydantic import ValidationError

from backend.arch.loader import SpecLoader
from backend.arch.spec_parsing import _parse_block_graph
from backend.arch.schema import BlockGraph, CacheSpec, GraphNode


def _declared_dtypes(graph):
    """Slot → the `dtype:` its cache entry declares, as written."""
    return {e.name: e.dtype for e in graph.cache.entries if e.dtype is not None}


def _attn_raw(**over):
    """A minimal attention component in authored form."""
    raw = {
        "role": "attention",
        "cache": {"family": "GQA", "entries": [
            {"node": "k", "kind": "kv"},
            {"node": "v", "kind": "kv", "dtype": "fp8"}]},
        "graph": {
            "k": {"op": "linear", "in": "x", "dims": {"in": "d_model", "out": "d_model"}},
            "v": {"op": "linear", "in": "x", "dims": {"in": "d_model", "out": "d_model"}},
            "out": {"op": "add", "in": ["k", "v"], "dim": "d_model"},
        },
        "outputs": ["out"],
    }
    raw.update(over)
    return raw


class TestCacheDeclaration(unittest.TestCase):
    """`cache:` is the one place a component says what it keeps between
    decode steps. The four fields it replaced — `kv_cached`,
    `kv_cache_dtype`, `kv_state_elements`, `kv_label` — survive only as
    read-only views (`kv_cached`, `kv_states`), so a cache fact has exactly
    one authoring site."""

    def test_entries_lower_to_the_cache_views(self):
        graph = _parse_block_graph(_attn_raw())
        self.assertEqual(graph.kv_label, "GQA")
        self.assertEqual(graph.kv_cached, ["k", "v"])
        self.assertEqual(_declared_dtypes(graph), {"v": "fp8"})
        self.assertEqual(graph.kv_states, [])
        self.assertEqual(graph.attn_display, "GQA")

    def test_recurrent_states_are_cache_entries(self):
        """A layer keeps as many fixed states as it has kernels streaming
        one — the scan's matrix and the causal conv's window — each named,
        shaped, and tied to its carrier."""
        graph = _parse_block_graph(_attn_raw(cache={
            "family": "GatedDeltaNet",
            "entries": [
                {"state": "recurrent_state", "kind": "recurrent_state", "carrier": "k",
                 "shape": ["n_heads", "d_head", "d_head"]},
                {"state": "conv_state", "kind": "conv_state", "carrier": "v", "dtype": "fp32",
                 "shape": ["kernel_size - 1", "C"]},
            ],
        }))
        self.assertEqual(
            [(e.state, e.carrier, e.shape) for e in graph.kv_states],
            [("recurrent_state", "k", ["n_heads", "d_head", "d_head"]),
             ("conv_state", "v", ["kernel_size - 1", "C"])])
        self.assertEqual(_declared_dtypes(graph), {"conv_state": "fp32"})
        self.assertEqual(graph.kv_cached, [])
        self.assertEqual(graph.kv_label, "GatedDeltaNet")

    def test_a_state_without_a_shape_is_refused_naming_the_fix(self):
        """No graph node outputs a state, so its axes are the one structural
        fact written about it; without them there is nothing to size it
        from, and the refusal says what to write."""
        with self.assertRaisesRegex(
                ValidationError,
                r"'recurrent_state' has no `shape:`.*outermost first.*"
                r"shape: \[n_lin_v_heads, d_lin_k_head, d_lin_v_head\]"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GatedDeltaNet",
                "entries": [{"state": "recurrent_state", "kind": "recurrent_state", "carrier": "k"}],
            }))

    def test_a_state_size_is_not_declared(self):
        """A state's size is the product of its axes; a declared element
        count is a second answer, refused with the shape to write instead."""
        with self.assertRaisesRegex(
                ValidationError,
                r"declares `elements: d_model`.*Replace `elements:` with "
                r"`shape: \[\.\.\.\]`"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GatedDeltaNet",
                "entries": [{"state": "recurrent_state", "kind": "recurrent_state", "carrier": "k",
                             "elements": "d_model"}],
            }))

    def test_a_state_has_at_least_one_axis(self):
        with self.assertRaisesRegex(ValidationError, "at least one axis"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GatedDeltaNet",
                "entries": [{"state": "recurrent_state", "kind": "recurrent_state", "carrier": "k",
                             "shape": []}],
            }))

    def test_a_state_without_a_carrier_is_rejected_with_the_fix(self):
        with self.assertRaisesRegex(ValidationError,
                                    r"names no `carrier:`.*add `carrier: "):
            _parse_block_graph(_attn_raw(cache={
                "family": "GatedDeltaNet",
                "entries": [{"state": "recurrent_state", "kind": "recurrent_state",
                             "shape": ["d_model"]}],
            }))

    def test_a_carrier_must_be_a_node_of_the_graph(self):
        with self.assertRaisesRegex(ValidationError,
                                    r"carrier 'scan', which is not a node"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GatedDeltaNet",
                "entries": [{"state": "recurrent_state", "kind": "recurrent_state", "carrier": "scan",
                             "shape": ["d_model"]}],
            }))

    def test_a_node_entry_takes_no_carrier(self):
        with self.assertRaisesRegex(ValidationError, "belongs to a `state:`"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GQA",
                "entries": [{"node": "k", "kind": "kv", "carrier": "k"}],
            }))

    def test_node_entry_refuses_a_declared_shape(self):
        """A graph node already has a shape; declaring one is a second
        answer that can disagree with the first."""
        with self.assertRaisesRegex(ValidationError, "node's resolved output"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GQA",
                "entries": [{"node": "k", "kind": "kv", "shape": ["d_model"]}],
            }))

    def test_entry_declares_exactly_one_storage_law(self):
        with self.assertRaisesRegex(ValidationError, "exactly one"):
            _parse_block_graph(_attn_raw(cache={
                "entries": [{"node": "k", "kind": "kv", "state": "s", "shape": ["d_model"]}],
            }))

    def test_slot_must_name_a_node_in_this_graph(self):
        with self.assertRaisesRegex(ValidationError, "not a node in this"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GQA", "entries": [{"node": "missing", "kind": "kv"}],
            }))

    def test_slot_may_reach_into_a_subcomponent_port(self):
        graph = _parse_block_graph(_attn_raw(
            cache={"family": "MLA",
                   "entries": [{"node": "indexer.k_norm", "kind": "index_key", "dtype": "fp8"}]},
            graph={
                "indexer": {"op": "dsv4_indexer", "in": "x"},
            },
            outputs=["indexer"],
        ))
        self.assertEqual(graph.kv_cached, ["indexer.k_norm"])
        self.assertEqual(_declared_dtypes(graph), {"indexer.k_norm": "fp8"})

    def test_duplicate_carriers_are_rejected(self):
        """One kernel streams one state; two states on it would put two
        tensors behind one row with nothing saying which read is which."""
        with self.assertRaisesRegex(ValidationError,
                                    r"carrier of both 'a' and 'b'"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GQA",
                "entries": [{"state": "a", "kind": "recurrent_state", "carrier": "k",
                             "shape": ["d_model"]},
                            {"state": "b", "kind": "recurrent_state", "carrier": "k",
                             "shape": ["d_model"]}],
            }))

    def test_duplicate_state_names_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "'a' is used twice"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GQA",
                "entries": [{"state": "a", "kind": "recurrent_state", "carrier": "k",
                             "shape": ["d_model"]},
                            {"state": "a", "kind": "recurrent_state", "carrier": "v",
                             "shape": ["d_model"]}],
            }))

    def test_a_state_may_not_share_a_node_slot_name(self):
        """The name keys the dtype map: a state `k` declared fp32 beside the
        node slot `k` would make the node slot fp32 too."""
        with self.assertRaisesRegex(
                ValidationError,
                r"'k' is used twice \(a node slot and a state\).*rename"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GQA",
                "entries": [{"node": "k", "kind": "kv"}, {"state": "k", "kind": "conv_state", "carrier": "v",
                                  "dtype": "fp32", "shape": ["d_model"]}],
            }))

    def test_a_node_slot_may_not_be_declared_twice(self):
        with self.assertRaisesRegex(ValidationError, "'k' is used twice"):
            _parse_block_graph(_attn_raw(cache={
                "family": "GQA", "entries": [{"node": "k", "kind": "kv"}, {"node": "k", "kind": "kv", "dtype": "fp8"}],
            }))

    def test_unknown_cache_key_is_rejected(self):
        with self.assertRaises(ValidationError):
            _parse_block_graph(_attn_raw(cache={
                "family": "GQA", "entries": [{"node": "k", "kind": "kv"}], "window": "s_window",
            }))

    def test_attention_role_requires_a_cache_family(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            _parse_block_graph(_attn_raw(cache={"entries": [{"node": "k", "kind": "kv"}]}))

    def test_attn_variant_repeating_the_family_is_rejected(self):
        """`attn_display` already falls back to the family, so a variant
        equal to it is a line that can only drift."""
        with self.assertRaisesRegex(ValidationError, "repeats cache.family"):
            _parse_block_graph(_attn_raw(attn_variant="GQA"))

    def test_attn_variant_that_differs_is_kept(self):
        graph = _parse_block_graph(_attn_raw(
            cache={"family": "MLA", "entries": [{"node": "k", "kind": "kv"}]}, attn_variant="DSA"))
        self.assertEqual(graph.kv_label, "MLA")
        self.assertEqual(graph.attn_display, "DSA")

    def test_cache_views_are_not_writable(self):
        """The views have no setter, so no code path can put a cache fact
        anywhere but `cache:`."""
        bg = BlockGraph(
            inputs=["x"],
            graph={"k": GraphNode(op="linear", inputs=["x"],
                                  dims={"in": "d_model", "out": "d_model"})},
            outputs=["k"],
            cache=CacheSpec(family="GQA", entries=[{"node": "k", "kind": "kv"}]),
        )
        with self.assertRaises(AttributeError):
            bg.kv_label = "MLA"


class TestCacheSlotKind(unittest.TestCase):
    """Every cache entry declares what it holds. A deployment chooses cache
    precision per kind of slot — an MLA layer's latent, rope key and indexer
    key are three different reads — and a node's name is not a declaration
    of which one it is."""

    def _entries(self, *entries):
        return _parse_block_graph(_attn_raw(
            cache={"family": "GQA", "entries": list(entries)}))

    def test_the_kind_lands_on_the_entry(self):
        graph = self._entries({"node": "k", "kind": "kv"},
                              {"node": "v", "kind": "index_key"})
        self.assertEqual([(e.name, e.kind) for e in graph.cache.entries],
                         [("k", "kv"), ("v", "index_key")])

    def test_an_entry_without_a_kind_is_refused_with_the_vocabulary(self):
        with self.assertRaisesRegex(
                ValidationError,
                r"'k' declares no `kind:`.*\['compressed_kv', 'index_key', "
                r"'kv', 'latent', 'rope_key'\]"):
            self._entries({"node": "k"})

    def test_a_bare_node_name_is_refused_naming_the_mapping_to_write(self):
        with self.assertRaisesRegex(
                ValidationError, r"'k' is a bare node name.*\{node: k, kind: "):
            self._entries("k")

    def test_a_kind_outside_the_vocabulary_is_refused(self):
        with self.assertRaisesRegex(ValidationError,
                                    r"`kind: key`, which is not a cache slot"):
            self._entries({"node": "k", "kind": "key"})

    def test_a_kind_lives_under_its_storage_law(self):
        """A per-token kind on a fixed state, or a state kind on a per-token
        node, says the slot grows the other way from how it is declared."""
        with self.assertRaisesRegex(ValidationError,
                                    r"`state:` entry and declares `kind: kv`"):
            self._entries({"state": "s", "kind": "kv", "carrier": "k",
                           "shape": ["d_model"]})
        with self.assertRaisesRegex(
                ValidationError, r"`node:` entry and declares `kind: conv_state`"):
            self._entries({"node": "k", "kind": "conv_state"})

    def test_every_library_entry_declares_a_kind_in_the_vocabulary(self):
        """Every component that keeps a cache — library, drafts, decode
        twins — declares a kind on each entry."""
        from backend.arch import cache_families
        root = Path(__file__).resolve().parents[3] / "backend/arch/components"
        import yaml
        seen = 0
        for path in sorted(root.rglob("*.yaml")):
            doc = yaml.safe_load(path.read_text())
            cache = (doc or {}).get("cache") if isinstance(doc, dict) else None
            if not cache:
                continue
            for entry in CacheSpec(**cache).entries:
                self.assertIn(entry.kind, cache_families.slot_kinds(), path)
                seen += 1
        self.assertGreater(seen, 100)


class TestDelegatedSubCacheKind(unittest.TestCase):
    """An entry naming a sub-component that keeps its own cache gives one
    kind to that sub-cache's per-token tensors. It may relabel one kind
    (DSv4's indexer pools its compressor's rows as an `index_key`); it may
    not collapse several into one."""

    def _spec(self, inner_kinds, outer_kind="index_key"):
        from backend.arch.schema import ArchSpec

        inner = BlockGraph(
            inputs=["x"],
            graph={"a": GraphNode(op="linear", inputs=["x"],
                                  dims={"in": "d_model", "out": "d_model"}),
                   "b": GraphNode(op="linear", inputs=["x"],
                                  dims={"in": "d_model", "out": "d_model"})},
            outputs=["a"],
            cache=CacheSpec(entries=[
                {"node": name, "kind": kind}
                for name, kind in zip(("a", "b"), inner_kinds, strict=True)]),
        )
        outer = BlockGraph(
            inputs=["x"],
            graph={"sub": GraphNode(op="inner", inputs=["x"])},
            outputs=["sub"],
            cache=CacheSpec(family="MLA", entries=[
                {"node": "sub", "kind": outer_kind}]),
        )
        return ArchSpec(name="t", model_id="t", hf_class="t",
                        dims={"d_model": "8"}, model=[],
                        blocks={"inner": inner, "outer": outer})

    def test_one_kind_beneath_may_be_relabelled(self):
        self._spec(["compressed_kv", "compressed_kv"])

    def test_two_kinds_beneath_one_label_are_refused(self):
        """A latent and a rope key under one `index_key` would price both at
        whatever the index key is given."""
        with self.assertRaisesRegex(
                ValidationError,
                r"'sub' names 'inner', whose own cache keeps per-token "
                r"tensors of 2 kinds.*`kind: index_key` would give them all "
                r"one.*\{node: sub\.<slot>"):
            self._spec(["latent", "rope_key"])


class TestTokenWindowTakesNoFormat(unittest.TestCase):
    """`token_window` holds integer ids: the vocabulary marks it
    `storage_format: false`, a precision choice naming it is refused, and an
    entry of it states its own dtype so the deployment's KV dtype never
    reaches it."""

    def test_the_vocabulary_marks_it(self):
        from backend.arch import cache_families
        self.assertFalse(cache_families.slot_kinds()["token_window"].storage_format)
        self.assertNotIn("token_window", cache_families.formattable_kinds())
        self.assertIn("recurrent_state", cache_families.formattable_kinds())

    def test_a_format_asked_of_it_is_refused(self):
        from backend.arch import cache_families
        cache_families.refuse_storage_format("conv_state")
        with self.assertRaisesRegex(ValueError,
                                    "'token_window' takes no storage format"):
            cache_families.refuse_storage_format("token_window")

    def test_an_entry_without_its_own_dtype_is_refused(self):
        from backend.arch.schema import CacheEntry
        CacheEntry(state="ids", kind="token_window", carrier="lookup",
                   shape=["2"], dtype="I64")
        with self.assertRaisesRegex(ValidationError,
                                    r"which no storage format applies to"):
            CacheEntry(state="ids", kind="token_window", carrier="lookup",
                       shape=["2"])


class TestCacheSlotDtype(unittest.TestCase):
    """A slot's `dtype:` is a spelling, or the config field a checkpoint
    states it in (`config.x ?? y`, as `dims:` bindings are written)."""

    def _entry(self, dtype):
        from backend.arch.schema import CacheEntry
        return CacheEntry(state="recurrent_state", kind="recurrent_state",
                          carrier="scan", shape=["d"], dtype=dtype)

    def _model(self, stated="BF16"):
        asked = []

        def model_dtype():
            asked.append(True)
            return stated
        return model_dtype, asked

    def test_a_spelling_is_kept_as_written(self):
        model, asked = self._model()
        self.assertEqual(self._entry("fp32").resolved_dtype({}, model), "fp32")
        self.assertEqual(asked, [])

    def test_a_config_field_resolves_to_its_dtype(self):
        model, asked = self._model()
        entry = self._entry("config.mamba_ssm_cache_dtype ?? model")
        self.assertEqual(entry.resolved_dtype(
            {"mamba_ssm_cache_dtype": "float32"}, model), "F32")
        self.assertEqual(asked, [])

    def test_the_chain_falls_through_to_the_model_precision(self):
        model, asked = self._model("BF16")
        entry = self._entry("config.mamba_ssm_cache_dtype ?? model")
        self.assertEqual(entry.resolved_dtype({}, model), "BF16")
        self.assertEqual(asked, [True])

    def test_nested_fields_are_tried_in_order_then_the_literal(self):
        model, _ = self._model()
        entry = self._entry(
            "config.text_config.mamba_ssm_dtype ?? config.mamba_ssm_dtype ?? F32")
        self.assertEqual(entry.resolved_dtype(
            {"text_config": {"mamba_ssm_dtype": "float32"}}, model), "F32")
        self.assertEqual(entry.resolved_dtype(
            {"mamba_ssm_dtype": "float32"}, model), "F32")
        self.assertEqual(entry.resolved_dtype({}, model), "F32")

    def test_a_config_that_contradicts_the_forward_is_refused(self):
        """The literal a chain ends on is the HF forward's dtype; a config
        stating another is a disagreement the record has to settle, not one
        the resolver settles by picking a side."""
        model, _ = self._model()
        entry = self._entry("config.mamba_ssm_cache_dtype ?? F32")
        with self.assertRaisesRegex(
                ValueError, r"'bfloat16' \(BF16\).*ends on F32.*HF forward"):
            entry.resolved_dtype({"mamba_ssm_cache_dtype": "bfloat16"}, model)

    def test_a_chain_with_no_fallback_and_no_field_is_refused(self):
        model, _ = self._model()
        with self.assertRaisesRegex(ValueError, "names no fallback"):
            self._entry("config.mamba_ssm_cache_dtype").resolved_dtype(
                {}, model)

    def test_a_model_fallback_when_the_model_states_none_is_refused(self):
        model, _ = self._model(None)
        with self.assertRaisesRegex(ValueError, "states none"):
            self._entry("config.x ?? model").resolved_dtype({}, model)

    def test_a_config_value_that_is_no_dtype_is_refused(self):
        model, _ = self._model()
        with self.assertRaisesRegex(ValueError, "'auto'.*not a dtype"):
            self._entry("config.x ?? model").resolved_dtype({"x": "auto"}, model)

    def test_malformed_bindings_are_refused_at_load(self):
        for text, message in (
                ("float33", "neither a config field"),
                ("model", "names no config field"),
                ("F32 ?? config.x", "every term before the last"),
                ("config.x ?? bogus", "neither a config field")):
            with self.subTest(text=text), \
                    self.assertRaisesRegex(ValidationError, message):
                self._entry(text)


class TestTheRecurrentStateReachesItsKernel(unittest.TestCase):
    """A `state:` entry declares bytes that no graph node outputs — that is
    what makes it a state entry rather than a node one. The byte walk finds
    every other cache stream's reader by following an edge back to what
    produced it; here there is no edge, so the entry names the kernel that
    streams it. Without that, a Mamba layer's decode — which is that read,
    almost entirely — came out free."""

    def _spec(self, op, operands=1, states=None, conv_dims=None, **kw):
        from backend.arch.schema import ArchSpec

        if states is None:
            states = [{"state": "recurrent_state", "kind": "recurrent_state", "carrier": "scan",
                       "shape": ["d_model", "d_model"]}]
        block = BlockGraph(
            inputs=["x"],
            graph={"scan": GraphNode(op=op, inputs=["x"] * operands,
                                     dims={"dim": "d_model"}),
                   "conv": GraphNode(op="conv1d", inputs=["x"],
                                     dims=conv_dims or {"C": "d_model",
                                                        "kernel_size": "4"})},
            outputs=["scan"],
            role="attention",
            cache=CacheSpec(family="Mamba2", entries=states),
        )
        return ArchSpec(name="t", model_id="t", hf_class="t",
                        dims={"d_model": "8"}, model=[],
                        blocks={"mixer": block}, **kw)

    def test_a_registered_recurrence_satisfies_it(self):
        self._spec("mamba2_scan", operands=7)

    def test_a_conv_may_carry_its_own_window(self):
        """The conv window is a second state, on the conv — `conv1d` is
        not a recurrence op, and naming it as a carrier is what ties it."""
        self._spec("mamba2_scan", operands=7, states=[
            {"state": "recurrent_state", "kind": "recurrent_state", "carrier": "scan",
             "shape": ["d_model", "d_model"]},
            {"state": "conv_state", "kind": "conv_state", "carrier": "conv",
             "shape": ["kernel_size - 1", "C"]},
        ])

    def test_a_misspelt_dim_in_a_state_is_rejected_with_the_fix(self):
        """A state axis resolves in the carrier's frame; a name nothing
        binds is a typo, caught at load rather than deep in the walk."""
        with self.assertRaisesRegex(
                ValidationError,
                r"reads \['kernel_sise'\].*carrier 'conv'.*Fix the spelling"):
            self._spec("mamba2_scan", operands=7, states=[
                {"state": "recurrent_state", "kind": "recurrent_state", "carrier": "scan",
                 "shape": ["d_model", "d_model"]},
                {"state": "conv_state", "kind": "conv_state", "carrier": "conv",
                 "shape": ["kernel_sise - 1", "C"]},
            ])

    def test_a_state_axis_counting_positions_is_refused(self):
        """A state is one fixed tensor per request: an axis made of the
        sequence makes it grow with the context, which is a per-token cache
        and a different storage law."""
        with self.assertRaisesRegex(
                ValidationError,
                r"'recurrent_state' axis 1 `s_ctx` reads \['s_ctx'\], which "
                r"count positions or requests.*`node:` entry"):
            self._spec("mamba2_scan", operands=7, states=[
                {"state": "recurrent_state", "kind": "recurrent_state", "carrier": "scan",
                 "shape": ["d_model", "s_ctx"]},
                {"state": "conv_state", "kind": "conv_state", "carrier": "conv",
                 "shape": ["kernel_size - 1", "C"]},
            ])

    def test_a_state_axis_counting_requests_is_refused_by_kind(self):
        """Refused by what the name counts, reached through the carrier's
        own `dims:` too: `C` bound to the batch is still the batch."""
        with self.assertRaisesRegex(
                ValidationError,
                r"'conv_state' axis 1 `C` reads \['B'\]"):
            self._spec("mamba2_scan", operands=7, states=[
                {"state": "recurrent_state", "kind": "recurrent_state", "carrier": "scan",
                 "shape": ["d_model", "d_model"]},
                {"state": "conv_state", "kind": "conv_state", "carrier": "conv",
                 "shape": ["kernel_size - 1", "C"]},
            ], conv_dims={"C": "B * d_model", "kernel_size": "4"})

    def test_functions_in_a_state_are_not_dims(self):
        self._spec("mamba2_scan", operands=7, states=[
            {"state": "recurrent_state", "kind": "recurrent_state", "carrier": "scan",
             "shape": ["Max(d_model, 4)", "d_model"]},
            {"state": "conv_state", "kind": "conv_state", "carrier": "conv",
             "shape": ["kernel_size - 1", "C"]},
        ])

    def test_a_recurrence_with_no_state_is_rejected_with_the_fix(self):
        """A carrier without a state: the op streams one by declaration, and
        nothing would bill it."""
        with self.assertRaisesRegex(
                ValidationError,
                r"node 'scan' runs 'mamba2_scan'.*no `cache.entries` state "
                r"names it.*\{state: <name>, carrier: scan"):
            self._spec("mamba2_scan", operands=7, states=[
                {"state": "conv_state", "kind": "conv_state", "carrier": "conv",
                 "shape": ["kernel_size - 1", "C"]}])

    def test_a_carrier_must_be_a_kernel(self):
        with self.assertRaisesRegex(ValidationError,
                                    "not a primitive or custom op"):
            self._spec("some_subcomponent")

    def test_a_component_scoped_recurrence_satisfies_it(self):
        """GLA declares its own kernel under the component's `custom_ops:`.
        Both kinds answer the same question, so both count."""
        from backend.arch.schema import CustomOpSpec

        self._spec("my_recurrence", custom_ops={
            "my_recurrence": CustomOpSpec(
                flops="0", params="0",
                signature={"x": ["B", "s_ctx", "d_model"]},
                couples_on=[], shape_kind="preserve",
                carries_recurrent_state=True, description="t"),
        })

    def test_a_component_scoped_recurrence_needs_its_state(self):
        from backend.arch.schema import CustomOpSpec

        with self.assertRaisesRegex(ValidationError, "no `cache.entries`"):
            self._spec("my_recurrence", states=[], custom_ops={
                "my_recurrence": CustomOpSpec(
                    flops="0", params="0",
                    signature={"x": ["B", "s_ctx", "d_model"]},
                    couples_on=[], shape_kind="preserve",
                    carries_recurrent_state=True, description="t"),
            })


class TestClaimsSidecar(unittest.TestCase):
    """Evidence lives beside the component, not inside it."""

    def test_shipped_component_picks_up_its_claims(self):
        graph = SpecLoader()._component_lib["gqa_attention"]
        self.assertEqual(graph.kv_label, "GQA")
        self.assertEqual(graph.kv_cached, ["k_rope", "v_split_heads"])
        self.assertIsNone(graph.attn_variant)
        self.assertEqual(graph.attn_display, "GQA")
        self.assertIsNotNone(graph.axioms)
        self.assertEqual(len(graph.axioms.checks), 1)

    def _tree(self, tmp):
        base = Path(tmp)
        (base / "components" / "library").mkdir(parents=True)
        (base / "components" / "adapters").mkdir()
        (base / "components" / "drafts").mkdir()
        (base / "claims" / "components").mkdir(parents=True)
        (base / "models").mkdir()
        return base

    def test_claims_are_loaded_from_component_source_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._tree(tmp)
            (base / "components" / "library" / "gqa_attention.yaml").write_text(
                textwrap.dedent("""\
                    role: attention
                    cache: { family: GQA, entries: [{ node: k, kind: kv }] }
                    graph:
                      k: { op: linear, in: x, dims: { in: d_model, out: d_model } }
                    outputs: [k]
                """)
            )
            (base / "claims" / "components" / "gqa_attention.yaml").write_text(
                textwrap.dedent("""\
                    schema: huggingarch.claims/v1
                    subject: gqa_attention
                    axioms:
                      checks:
                        - { kind: kv_growth, storage: per_token, expect: d_model }
                """)
            )

            graph = SpecLoader(base_dir=str(base))._component_lib["gqa_attention"]
            self.assertEqual(graph.kv_label, "GQA")
            self.assertIsNotNone(graph.axioms)
            self.assertEqual(graph.axioms.checks[0].expect, "d_model")

    def test_an_overlay_model_inheriting_an_inline_block_keeps_its_claims(self):
        """Claims are found by the component's identity, not by where the
        spec reaching it was written: a thin overlay model that inherits a
        library model's inline block through `base:` is judged against the
        claims the library states about that block."""
        import json
        arch = Path(__file__).resolve().parents[3] / "backend" / "arch"
        config = json.loads(
            (arch / "snapshots" / "XiaomiMiMo__MiMo-V2.5.json").read_text())["config"]
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            models.mkdir()
            (models / "test__mimo-thin.yaml").write_text(
                "base: mimo_v2_5\nmatch:\n  model_id: [test/mimo-thin]\n")
            loader = SpecLoader(overlay_model_dirs=[str(models)])
            resolved = loader.resolve(loader.get("test__mimo-thin"), config)
        claims = resolved.spec.get_block("mimo_v2_5_audio_attention").axioms
        self.assertIsNotNone(claims)
        self.assertEqual([c.storage for c in claims.checks], ["none"])

    def test_inline_axioms_are_rejected(self):
        """Claims are declared in one place, `claims/components/`; a block
        inside the component would be a second copy that can disagree with
        the one the validator reads."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._tree(tmp)
            (base / "components" / "library" / "dup.yaml").write_text(
                textwrap.dedent("""\
                    role: attention
                    cache: { family: GQA, entries: [{ node: k, kind: kv }] }
                    axioms:
                      checks:
                        - { kind: kv_growth, storage: per_token, expect: d_model }
                    graph:
                      k: { op: linear, in: x, dims: { in: d_model, out: d_model } }
                    outputs: [k]
                """)
            )
            (base / "claims" / "components" / "dup.yaml").write_text(
                textwrap.dedent("""\
                    schema: huggingarch.claims/v1
                    subject: dup
                    axioms:
                      checks:
                        - { kind: kv_growth, storage: per_token, expect: d_model }
                """)
            )

            loader = SpecLoader(base_dir=str(base))
            self.assertNotIn("dup", loader._component_lib)


if __name__ == "__main__":
    unittest.main()
