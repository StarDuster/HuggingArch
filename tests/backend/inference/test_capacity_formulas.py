"""Capacity popover formulas are display. Grouping and the payload keys
are the contract; the numbers they explain stay in capacity.py."""
from types import SimpleNamespace

from backend.inference.capacity_formulas import capacity_formulas


class _Reader:
    """A cache view whose per-request formula is 8 B per token at 128."""

    def request_formula(self, cache_len):
        return {"label": "Per-request KV", "symbolic": "(n_layers × bpe) × s_kv",
                "substituted": f"(8) × {cache_len}",
                "result_bytes": 8.0 * cache_len, "subterms": []}


def test_capacity_formulas_payload_keys():
    preparation = SimpleNamespace(
        mem_total_bytes=80e9,
        gpu_memory_fraction=0.9,
        host_offload=False, host_offload_bytes=(0.0,),
        usable_bytes=80e9 * 0.9,
        parallel=SimpleNamespace(dp_attention=1, dp_replicas=1),
    )
    ceilings = SimpleNamespace(
        kv_reader=_Reader(),
        per_token_per_rank=8.0,
        kv_per_class_breakdown=None,
        kv_per_rank={},
        request_tokens=128,
        effective_storage_tokens=128,
        per_request_kv_bytes=1024.0,
        weight_bytes_per_rank=1e9,
        kv_budget_bytes=70e9,
        kv_static_per_rank=0.0,
        kv_constant_state_per_rank=0.0,
        max_tokens=1000,
        max_tokens_total=1000,
        max_batch=8,
        max_batch_total=8,
        max_prefill_batch=4,
        overhead_bytes=0.0,
        usable_for_tokens=70e9,
        peak_act_per_b1=1e6,
        fits=True,
        weight_deficit_bytes=0.0,
    )
    out = capacity_formulas(
        preparation=preparation, ceilings=ceilings,
        weight_breakdown={"groups": [], "model_level": [],
                          "deployment_data_bytes": 0.0,
                          "deployment_scale_bytes": 0.0},
    )
    assert set(out) == {
        "mem_total", "usable", "deployment",
        "kv_budget", "max_tokens", "max_tokens_total", "max_batch",
        "max_batch_total", "max_prefill_batch", "per_request_kv", "weight_per_rank",
    }
    assert out["mem_total"]["result_bytes"] == 80e9
    assert out["usable"]["result_bytes"] == 72e9
    assert out["usable"]["symbolic"] == "mem_total × gpu_memory_fraction"
    assert out["deployment"]["result_bytes"] == 0


def test_kv_budget_formula_is_zero_when_weights_do_not_fit():
    """The published budget is 0 when the rank does not fit; the shortfall
    rides `weight_deficit`, so a page does not subtract the two fields."""
    preparation = SimpleNamespace(
        mem_total_bytes=80e9,
        gpu_memory_fraction=0.9,
        host_offload=False, host_offload_bytes=(0.0,),
        usable_bytes=80e9 * 0.9,
        parallel=SimpleNamespace(dp_attention=1, dp_replicas=1),
    )
    ceilings = SimpleNamespace(
        kv_reader=_Reader(),
        per_token_per_rank=8.0,
        kv_per_class_breakdown=None,
        kv_per_rank={},
        request_tokens=128,
        effective_storage_tokens=128,
        per_request_kv_bytes=1024.0,
        weight_bytes_per_rank=100e9,
        kv_budget_bytes=0.0,
        weight_deficit_bytes=28e9,
        kv_static_per_rank=0.0,
        kv_constant_state_per_rank=0.0,
        max_tokens=0,
        max_tokens_total=0,
        max_batch=0,
        max_batch_total=0,
        max_prefill_batch=0,
        overhead_bytes=0.0,
        usable_for_tokens=0.0,
        peak_act_per_b1=1e6,
        fits=False,
    )
    out = capacity_formulas(
        preparation=preparation, ceilings=ceilings,
        weight_breakdown={"groups": [], "model_level": [],
                          "deployment_data_bytes": 0.0,
                          "deployment_scale_bytes": 0.0},
    )
    assert out["kv_budget"]["result_bytes"] == 0.0
    assert out["kv_budget"]["substituted"].endswith("→ 0")
    assert out["weight_deficit"]["result_bytes"] == 28e9


def _prep_ceilings(**wb_extra):
    preparation = SimpleNamespace(
        mem_total_bytes=80e9,
        gpu_memory_fraction=0.9,
        host_offload=False, host_offload_bytes=(0.0,),
        usable_bytes=80e9 * 0.9,
        parallel=SimpleNamespace(dp_attention=1, dp_replicas=1),
    )
    ceilings = SimpleNamespace(
        kv_reader=_Reader(),
        per_token_per_rank=8.0,
        kv_per_class_breakdown=None,
        kv_per_rank={},
        request_tokens=128,
        effective_storage_tokens=128,
        per_request_kv_bytes=1024.0,
        weight_bytes_per_rank=1e9,
        kv_budget_bytes=70e9,
        kv_static_per_rank=0.0,
        kv_constant_state_per_rank=0.0,
        max_tokens=1000,
        max_tokens_total=1000,
        max_batch=8,
        max_batch_total=8,
        max_prefill_batch=4,
        overhead_bytes=0.0,
        usable_for_tokens=70e9,
        peak_act_per_b1=1e6,
        fits=True,
        weight_deficit_bytes=0.0,
    )
    wb = {"groups": [], "model_level": [], "deployment_data_bytes": 0.0,
          "deployment_scale_bytes": 0.0, **wb_extra}
    return preparation, ceilings, wb


def test_deployment_formula_splits_data_and_scale():
    """The Memory card's data / scale addends are named subterms, so the
    page can hover each number without guessing an index."""
    preparation, ceilings, wb = _prep_ceilings(
        groups=[{
            "data_bytes_full": 1000.0,
            "scale_bytes_full": 50.0,
        }],
        model_level=[{
            "data_bytes_full": 200.0,
            "scale_bytes_full": 10.0,
        }],
        deployment_total_bytes=1260.0,
        deployment_data_bytes=1200.0,
        deployment_scale_bytes=60.0,
    )
    out = capacity_formulas(
        preparation=preparation, ceilings=ceilings, weight_breakdown=wb,
    )
    dep = out["deployment"]
    assert dep["result_bytes"] == 1260.0
    assert dep["substituted"] == "1200 + 60"
    by_sym = {st["symbol"]: st for st in dep["subterms"]}
    assert by_sym["data"]["result_bytes"] == 1200.0
    assert by_sym["scale"]["result_bytes"] == 60.0


def test_scale_no_leaf_claims_is_its_own_declared_term():
    """Scale metadata with no parent leaf is the breakdown's declared third
    term of the deployment total — named from that field, not a residual."""
    preparation, ceilings, wb = _prep_ceilings(
        groups=[{"data_bytes_full": 1000.0, "scale_bytes_full": 50.0}],
        deployment_total_bytes=1055.0,
        deployment_data_bytes=1000.0,
        deployment_scale_bytes=50.0,
        unattributed_scale_bytes=5.0,
    )
    dep = capacity_formulas(
        preparation=preparation, ceilings=ceilings, weight_breakdown=wb,
    )["deployment"]
    assert dep["substituted"] == "1000 + 50 + 5"
    assert [st["result_bytes"] for st in dep["subterms"]] == [1000.0, 50.0, 5.0]


def test_per_request_kv_is_the_readers_formula_per_gpu():
    """The popover is the cache view's own formula at the request length —
    the object that produced `per_request_kv_bytes` — labelled per GPU."""
    preparation, ceilings, wb = _prep_ceilings()
    tree = capacity_formulas(
        preparation=preparation, ceilings=ceilings, weight_breakdown=wb,
    )["per_request_kv"]
    assert tree["label"] == "Per-request KV / GPU"
    assert tree["result_bytes"] == 1024.0
    assert tree["substituted"] == "(8) × 128"
