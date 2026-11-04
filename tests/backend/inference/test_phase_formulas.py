"""Phase popover formulas are display. Parent rows get Σ children
from the same tree invariants already sum; this does not add rows."""
from backend.inference.phase_formulas import (
    _fmt_si,
    _t_comm_formula_for_row,
    _t_compute_formula_for_row,
    _t_memory_formula_for_row,
    attach_parent_formulas,
    enrich_module_breakdown,
)


def test_fmt_si_letter_units_concatenate():
    assert _fmt_si(100350, " B") == "100.35 KB"
    assert _fmt_si(100350, "B") == "100.35 KB"
    assert _fmt_si(500, " B") == "500 B"
    assert _fmt_si(1_230_000, " FLOPs") == "1.23 M FLOPs"


def test_attach_parent_formulas_sums_immediate_children():
    rows = [
        {"name": "attn_block", "op": "<template>", "depth": 0,
         "flops_per_rank": 30},
        {"name": "self_attn", "op": "wrapper", "depth": 1,
         "flops_per_rank": 10},
        {"name": "q_proj", "op": "linear", "depth": 2,
         "flops_per_rank": 10, "flops_formula": {"symbolic": "leaf"}},
        {"name": "mlp", "op": "wrapper", "depth": 1,
         "flops_per_rank": 20},
        {"name": "up_proj", "op": "linear", "depth": 2,
         "flops_per_rank": 20, "flops_formula": {"symbolic": "leaf"}},
    ]
    attach_parent_formulas(
        rows, value_key="flops_per_rank", formula_key="flops_formula",
    )
    assert rows[2]["flops_formula"]["symbolic"] == "leaf"
    assert rows[0]["flops_formula"]["symbolic"] == "Σ 2 components"
    # The generic formula line: terms large→small, then the total.
    assert rows[0]["flops_formula"]["substituted"] == "mlp(20) + self_attn(10) = 30"
    assert rows[0]["flops_formula"]["result"] == "30"
    assert rows[1]["flops_formula"]["symbolic"] == "Σ 1 component"
    assert len(rows) == 5


def test_attach_parent_formulas_skips_existing():
    rows = [
        {"name": "block", "depth": 0, "flops_full": 3,
         "flops_formula": {"symbolic": "kept"}},
        {"name": "leaf", "depth": 1, "flops_full": 3},
    ]
    attach_parent_formulas(
        rows, value_key="flops_full", formula_key="flops_formula",
    )
    assert rows[0]["flops_formula"]["symbolic"] == "kept"


def test_t_term_formulas_divide_the_way_the_cell_was_priced():
    """Each roofline term's popover divides the cell's own numerator by the
    peak the time was priced at — the compute one names that peak's dtype,
    the memory one decomposes access_bytes so its parts sum to the divided
    figure."""
    f = _t_compute_formula_for_row(
        t_compute_s=2e-4, flops_per_rank=197_800_000_000,
        peak_flops_per_sec=989e12, act_dtype="BF16")
    assert f["symbolic"] == "t_compute = FLOPs / peak_flops"
    assert f["substituted"] == "197,800,000,000 / 9.89e+14"
    assert f["result"] == "200.0 µs"
    assert "989.00 T FLOP/s" in f["note"] and "BF16" in f["note"]
    m = _t_memory_formula_for_row(
        t_memory_s=1e-3, access_per_rank=3_350_000_000,
        peak_hbm_bw_bytes_per_sec=3.35e12,
        weight_bytes_step_pr=3_000_000_000,
        act_in_bytes_step_pr=350_000_000)
    assert m["symbolic"] == "t_memory = access_bytes / peak_bw"
    assert m["substituted"] == "3,350,000,000 / 3.35e+12"
    assert "weights=3.00 GB" in m["note"] and "act_in=350.00 MB" in m["note"]
    # A row that runs no compute / moves nothing publishes no formula.
    assert _t_compute_formula_for_row(
        t_compute_s=0.0, flops_per_rank=0,
        peak_flops_per_sec=989e12) is None
    assert _t_memory_formula_for_row(
        t_memory_s=0.0, access_per_rank=0,
        peak_hbm_bw_bytes_per_sec=3.35e12) is None


def test_t_term_formulas_divide_the_cells_own_numbers():
    """A routed leaf's FLOPs and bytes are its busiest rank's already
    (`layout.physical`): the popover divides the published numbers by the
    peaks, with nothing multiplied beside them."""
    f = _t_compute_formula_for_row(
        t_compute_s=247_250_000_000 / 989e12,
        flops_per_rank=247_250_000_000, peak_flops_per_sec=989e12,
        act_dtype="BF16")
    assert f["symbolic"] == "t_compute = FLOPs / peak_flops"
    assert f["substituted"] == "247,250,000,000 / 9.89e+14"
    assert f["result"] == "250.0 µs"
    m = _t_memory_formula_for_row(
        t_memory_s=3_437_500_000 / 3.35e12,
        access_per_rank=3_437_500_000, peak_hbm_bw_bytes_per_sec=3.35e12,
        weight_bytes_step_pr=3_000_000_000,
        act_in_bytes_step_pr=437_500_000)
    assert m["symbolic"] == "t_memory = access_bytes / peak_bw"
    assert m["substituted"] == "3,437,500,000 / 3.35e+12"


def test_enrich_fills_parent_t_term_formulas():
    """Aggregator rows sum their children's roofline terms; each parent's
    t_compute / t_memory popover is that Σ, a leaf keeps its own."""
    phase = {
        "flops_by_module": [
            {"name": "block", "op": "<template>", "depth": 0,
             "aggregation_id": "", "flops_per_rank": 10},
            {"name": "q_proj", "op": "linear", "depth": 1,
             "aggregation_id": "block|q_proj", "flops_per_rank": 10,
             "t_op_s": 0.002, "t_compute_s": 0.0015, "t_memory_s": 0.002,
             "access_bytes_per_rank": 100,
             "t_compute_formula": {"symbolic": "leaf compute"},
             "t_memory_formula": {"symbolic": "leaf memory"}},
        ],
    }
    enrich_module_breakdown(phase)
    parent, leaf = phase["flops_by_module"]
    assert parent["t_compute_formula"]["symbolic"].startswith("Σ")
    assert "q_proj" in parent["t_compute_formula"]["substituted"]
    assert parent["t_memory_formula"]["symbolic"].startswith("Σ")
    assert leaf["t_compute_formula"]["symbolic"] == "leaf compute"
    assert leaf["t_memory_formula"]["symbolic"] == "leaf memory"


def test_enrich_rolls_up_native_comm_and_fills_parent_t_op_formula():
    """Rows arrive complete (comm is a native tree row, timing already
    merged); enrich rolls the columns up and fills parent popovers."""
    phase = {
        "flops_by_module": [
            {"name": "attn_block", "op": "<template>", "depth": 0,
             "aggregation_id": "", "flops_per_rank": 10},
            {"name": "self_attn", "op": "wrapper", "depth": 1,
             "aggregation_id": "attn_block|self_attn", "flops_per_rank": 10},
            {"name": "q_proj", "op": "linear", "depth": 2,
             "aggregation_id": "attn_block|self_attn.q_proj",
             "flops_per_rank": 10,
             "t_op_s": 0.002, "t_comm_s": 0,
             "access_bytes_per_rank": 100, "comm_bytes_per_rank": 0,
             "t_op_formula": {"symbolic": "leaf"}},
            {"name": "all_reduce", "op": "all_reduce", "depth": 2,
             "aggregation_id": "attn_block|self_attn.all_reduce",
             "is_comm": True,
             "flops_per_rank": 0,
             "t_op_s": 0.001, "t_comm_s": 0.001,
             "access_bytes_per_rank": 0, "comm_bytes_per_rank": 50,
             "t_op_formula": {"symbolic": "ring"},
             "comm_world": 8, "comm_bw_label": "NVLink",
             "comm_bw_bytes_per_sec": 900e9,
             "comm_phases": [{"name": "intra"}]},
        ],
    }
    enrich_module_breakdown(phase)
    rows = phase["flops_by_module"]
    parent = next(r for r in rows if r["name"] == "self_attn")
    assert parent["t_op_formula"]["symbolic"].startswith("Σ")
    assert "all_reduce" in parent["t_op_formula"]["substituted"]
    assert "q_proj" in parent["t_op_formula"]["substituted"]
    # The parent's own cells roll up its subtree, comm included.
    assert parent["t_op_agg"] == 0.003
    assert parent["comm_bytes_agg"] == 50
    leaf = next(r for r in rows if r["name"] == "q_proj")
    assert leaf["t_op_formula"]["symbolic"] == "leaf"
    comm = next(r for r in rows if r.get("is_comm"))
    assert comm["comm_bw_label"] == "NVLink"
    assert comm["comm_world"] == 8
    assert comm["comm_phases"] == [{"name": "intra"}]


def test_enrich_parent_formula_names_only_the_rows_present():
    """With the fused tree, absorbed children are simply not rows — the
    parent formula is built from the rows the walk emitted."""
    phase = {
        "flops_by_module": [
            {"name": "self_attn", "op": "wrapper", "depth": 0,
             "aggregation_id": "attn|self_attn", "flops_per_rank": 20},
            {"name": "attn_apply", "op": "attn_apply", "depth": 1,
             "aggregation_id": "attn|self_attn.attn_apply",
             "flops_per_rank": 20,
             "t_op_s": 0.002, "t_comm_s": 0, "access_bytes_per_rank": 20,
             "comm_bytes_per_rank": 0},
        ],
    }
    enrich_module_breakdown(phase)
    parent = phase["flops_by_module"][0]
    assert "qk_score" not in parent["t_op_formula"]["substituted"]
    assert "attn_apply" in parent["t_op_formula"]["substituted"]
    assert "qk_score" not in parent["flops_formula"]["substituted"]
    assert parent["flops_formula"]["result"] == "20"
    assert parent["flops_formula"]["substituted"] == "attn_apply(20) = 20"


def test_phase_summary_formulas_expand_top_components():
    """The Summary rows' FLOPs / t_SOL popovers read like the module rows':
    the depth-0 component sum, not a bare `Σ …` with only a total. mem_read
    stays total-only — per-op access is read+write combined, so the tree
    cannot reconcile with the phase's reads figure."""
    phase = {
        "flops_per_step": 130, "mem_read_bytes": 999, "t_sol_s": 0.003,
        "formulas": {
            "flops": {"symbolic": "engine: Σ ops",
                      "substituted": "per-rank = Σ ops"},
            "mem_read": {"symbolic": "Σ leaf (…)",
                         "substituted": "= 999 bytes"},
            "t_sol": {"symbolic": "Σ_leaf max(…)",
                      "substituted": "= 0.003 s"},
        },
        "flops_by_module": [
            {"name": "dense", "op": "<template>", "depth": 0,
             "template_name": "dense", "layer_count": 3,
             "flops_per_rank": 30, "t_op_s": 0.001},
            {"name": "moe", "op": "<template>", "depth": 0,
             "template_name": "moe", "layer_count": 58,
             "flops_per_rank": 90, "t_op_s": 0.002},
            {"name": "lm_head", "op": "linear", "depth": 0,
             "flops_per_rank": 10, "t_op_s": 0.0},
        ],
    }
    enrich_module_breakdown(phase)
    f = phase["formulas"]["flops"]
    assert f["substituted"] == "moe(90) + dense(30) + lm_head(10) = 130"
    assert f["result"] == "130"
    t = phase["formulas"]["t_sol"]
    assert t["substituted"] == "moe(2.0 ms) + dense(1.0 ms) = 3.0 ms"
    assert t["result"] == "3.0 ms"
    assert phase["formulas"]["mem_read"]["substituted"] == "= 999 bytes"


def test_a_pipelined_step_time_is_not_expanded_over_one_stage():
    """At pp>1 the step time is every stage's plus the hops while the fold is
    the bound stage's rows: the FLOPs (that stage's) expand over the fold,
    the time keeps its own line rather than a sum that does not add up."""
    phase = {
        "flops_per_step": 100, "t_sol_s": 0.02,
        "pipeline_time": {"stages": 2, "t_sol": 0.02},
        "formulas": {
            "flops": {"symbolic": "s", "substituted": "x"},
            "t_sol": {"symbolic": "s", "substituted": "x"},
        },
        "flops_by_module": [
            {"name": "moe", "op": "<template>", "depth": 0,
             "template_name": "moe", "layer_count": 29,
             "flops_per_rank": 100, "t_op_s": 0.005},
        ],
    }
    enrich_module_breakdown(phase)
    assert phase["formulas"]["flops"]["substituted"] == "moe(100) = 100"
    assert phase["formulas"]["t_sol"]["substituted"] == "x"


def test_component_lines_sort_desc_and_cap_at_20():
    """Formula component terms read large→small and stop at 20; the Σ, the
    total, and the residual keep counting everything the cap dropped."""
    rows = [{"name": "block", "op": "<template>", "depth": 0,
             "flops_per_rank": 0}]
    rows += [{"name": f"m{i}", "op": "linear", "depth": 1,
              "flops_per_rank": i} for i in range(1, 26)]  # 25 children
    attach_parent_formulas(
        rows, value_key="flops_per_rank", formula_key="flops_formula",
    )
    f = rows[0]["flops_formula"]
    assert rows[0]["flops_formula"]["symbolic"] == "Σ 25 components"
    assert f["substituted"].startswith("m25(25) + m24(24) + m23(23)")
    assert f["substituted"].endswith("m6(6) = 325")  # top 20 of 25 terms
    assert f["note"] == "+ 5 smaller terms"
    assert f["result"] == "325"  # Σi for 1..25 — the cap drops nothing
    # 21 components → 20 shown + 1 more; the total still counts all 21.
    rows2 = [{"name": "block", "op": "<template>", "depth": 0,
              "flops_per_rank": 0}]
    rows2 += [{"name": f"c{i}", "op": "linear", "depth": 1,
               "flops_per_rank": 1} for i in range(21)]
    attach_parent_formulas(
        rows2, value_key="flops_per_rank", formula_key="flops_formula",
    )
    f2 = rows2[0]["flops_formula"]
    assert f2["note"] == "+ 1 smaller terms"
    assert f2["result"] == "21"


def test_phase_components_sort_desc():
    phase = {
        "flops_per_step": 60, "t_sol_s": None,
        "formulas": {"flops": {"symbolic": "s", "substituted": "x"}},
        "flops_by_module": [
            {"name": "small", "op": "linear", "depth": 0,
             "flops_per_rank": 5},
            {"name": "big", "op": "<template>", "depth": 0,
             "template_name": "big", "layer_count": 58,
             "flops_per_rank": 50},
            {"name": "mid", "op": "linear", "depth": 0,
             "flops_per_rank": 5},
        ],
    }
    enrich_module_breakdown(phase)
    f = phase["formulas"]["flops"]
    assert f["substituted"] == "big(50) + small(5) + mid(5) = 60"
    assert f["result"] == "60"


def test_a_hierarchical_collective_times_each_hop_on_its_fabric():
    """The t_comm popover is one subterm per fabric hop, each at one
    invocation of the row's `cnt`, and the row's time as the result."""
    phases = [
        {"name": "rs", "fabric": "intra", "fabric_label": "NVLink", "world": 8,
         "bytes_per_rank": 2.0e9, "bw_bytes_per_sec": 4.0e11, "t_phase_s": 0.005,
         "symbolic": "(n-1)/n * T", "substituted": "7/8 * (1e9)"},
        {"name": "ar", "fabric": "inter", "fabric_label": "RDMA", "world": 2,
         "bytes_per_rank": 1.0e9, "bw_bytes_per_sec": 5.0e10, "t_phase_s": 0.02,
         "symbolic": "2*(n-1)/n * T", "substituted": "1 * (1e9)"},
    ]
    f = _t_comm_formula_for_row(
        t_comm_s=0.025, comm_bytes_per_rank=3.0e9, comm_bw_bytes_per_sec=5.0e10,
        comm_bw_label="RDMA", comm_world=16, comm_bytes_formula=None,
        comm_phases=phases, cnt=10)
    assert f["symbolic"] == "t_comm = t_rs + t_ar"
    assert f["result"] == "25.0 ms"
    assert f["note"] == "world=16"
    assert [t["label"] for t in f["subterms"]] == [
        "rs (NVLink, world=8)", "ar (RDMA, world=2)"]
    assert f["subterms"][1]["result"] == "100.00 MB / 50.00 GB/s = 2.0 ms"


def test_a_collective_that_moves_nothing_has_no_t_comm_popover():
    assert _t_comm_formula_for_row(
        t_comm_s=0.0, comm_bytes_per_rank=0.0, comm_bw_bytes_per_sec=5.0e10,
        comm_bw_label="RDMA", comm_world=1, comm_bytes_formula=None,
        comm_phases=[], cnt=1) is None


def test_intensity_formula_divides_the_published_operands():
    from backend.inference.intensity import (
        arithmetic_intensity, intensity_formula)
    f = intensity_formula(6_000, 1_500)
    assert f["substituted"] == "6,000 / 1,500"
    assert f["result"] == f"{arithmetic_intensity(6_000, 1_500):.4g} FLOPs/B"
    # No bytes, no ratio — and no popover beside the "—".
    assert intensity_formula(6_000, 0) is None


def test_enrich_publishes_intensity_formula_on_the_subtree_operands():
    phase = {"flops_by_module": [
        {"name": "mlp", "op": "wrapper", "depth": 0, "flops_per_rank": 30,
         "access_bytes_per_rank": 0},
        {"name": "up_proj", "op": "linear", "depth": 1, "flops_per_rank": 30,
         "access_bytes_per_rank": 10},
    ]}
    enrich_module_breakdown(phase)
    parent, leaf = phase["flops_by_module"]
    assert parent["arithmetic_intensity"] == 3.0
    assert parent["arithmetic_intensity_formula"]["substituted"] == "30 / 10"
    assert leaf["arithmetic_intensity_formula"]["result"] == "3 FLOPs/B"
