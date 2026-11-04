"""An MXFP4 MoE expert GEMM is keyed at the activation its card's kernel eats.

A 4-bit expert bank is not served the same way on every card, and the
difference is the ACTIVATION operand, so it is a difference of cells:

* On SM100 the audited vLLM auto selector tries FlashInfer TRT-LLM before
  DeepGEMM, and that launch eats MXFP8: FP8 E4M3 over group 32 with UE8M0
  scales (`fused_moe/oracle/mxfp4.py:425-462`). This is the default for both
  DeepSeek activation requests — V4's group-128 checkpoint and V4.1's
  group-32 one.
* DeepGEMM's `m_grouped_fp8_fp4_gemm_nt_contiguous` does take FP8 over group
  128 against MXFP4 and is Blackwell-only, but that fact alone does not select
  it. The explicit `deep_gemm` / no-FlashInfer deployment fact is not in an
  invocation, so its provider stays disabled and that cell stays a selector
  gap.
* On SM90 vLLM's own dispatch answers the same checkpoint weight-only:
  Humming moves ahead of Marlin at compute capability 90 exactly
  (`quantization/utils/humming/priority.py:11`) and the default oracle hands
  it no activation key
  (Marlin asserts W4A16 at `fused_moe/experts/marlin_moe.py:569-576` and
  passes `hidden_states` through at `:122-129`; Humming's default input
  schema at `oracle/mxfp4.py:1546-1548`). sglang agrees — its SM90 entry
  point is "W4A16 by default" (`layers/quantization/mxfp4.py:416-418`,
  `:428-437`).

So a DeepSeek W4A8 model is one cell on an H100 and another on a B200, and
the plan that emits the cell has to say which. MiMo's Blackwell activation
treatment and DeepGEMM's non-default Blackwell launch are deliberately left
visible gaps rather than asserted.

The provider gates below are the other end of the same statement — the
H100-keyed cell has a kernel on an H100, and the Blackwell-keyed one is
still refused there for the reason that is true (its card), so it stays a
real gap in the coverage denominator rather than becoming measurable by
being mis-keyed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kernel_bench.box import BoxEnvs, find_box_profile
from kernel_bench.plan import (build_jobs_with_unplaced, gaps_by_signature,
                               gaps_path, read_jobs_jsonl, write_plan)

ENVS_DIR = str(ROOT / "kernel_bench" / "envs")

#: The corpus's MXFP4 MoEs whose checkpoint declares an activation
#: quantiser, so the card decides whether that quantiser is reached:
#: DeepSeek's dot-suffix scheme (`quant_schemes/mxfp4-dot-suffix.yaml`,
#: FP8 activation from the model's own `inference/model.py`) and MiMo's
#: compressed-tensors one (`mxfp4-pack-quantized.yaml` with
#: `input_activations: fp8_dynamic_group32`). First is the model the
#: census counted the most mis-keyed cells of.
QUANTISING = ["deepseek-ai/DeepSeek-V4.1-Flash",
              "deepseek-ai/DeepSeek-V4-Flash",
              "XiaomiMiMo/MiMo-V2.6-Pro-RL"]
BLACKWELL_UNRESOLVED = ["XiaomiMiMo/MiMo-V2.6-Pro-RL"]


def _moe_activations(model_id: str, gpu_sku: str) -> set:
    """The activation slot of every MXFP4 `grouped_gemm` cell a plan for
    `model_id` on `gpu_sku` emits — read off the emitted jobs, not off the
    precision resolver, so the assertion does not share its subject's
    premise."""
    from backend.inference.framework_bindings import (ServingDeployment,
                                                      get_binding)
    from kernel_bench.plan import build_jobs

    binding = get_binding("vllm").deployed(
        ServingDeployment.at(gpu_sku=gpu_sku, activation_dtype="BF16"))
    jobs, _dropped = build_jobs(
        model_id=model_id, batch_size=8, seq_len=1024, phase="decode",
        tp=8, framework="vllm", binding=binding, avg_output_tokens=512,
        gpu=gpu_sku)
    return {job["dtype_env"].get("activation")
            for job in jobs
            if job.get("signature") == "grouped_gemm"
            and "fp4" in str(job["dtype_env"].get("weight", ""))}


def _moe_jobs(model_id: str, gpu_sku: str):
    from backend.inference.framework_bindings import (ServingDeployment,
                                                      get_binding)
    from kernel_bench.plan import build_jobs

    binding = get_binding("vllm").deployed(
        ServingDeployment.at(gpu_sku=gpu_sku, activation_dtype="BF16"))
    return build_jobs(
        model_id=model_id, batch_size=8, seq_len=1024, phase="decode",
        tp=8, framework="vllm", binding=binding, avg_output_tokens=512,
        gpu=gpu_sku)[0]


@pytest.mark.parametrize("model_id", QUANTISING)
def test_sm90_keys_an_mxfp4_moe_weight_only(model_id):
    """The framework's pre-Blackwell launch declaration is W4A16, so the
    activation is the edge's on an H100. This is not a missing-FP4 fallback
    saving an inferred FP8 launch: the launch and its BF16 MMA family are
    stated together, and an FP8 key there still names a launch the default
    does not take."""
    assert _moe_activations(model_id, "H100-SXM") == {"bf16"}


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-V4-Flash",
                                      "deepseek-ai/DeepSeek-V4-Pro",
                                      "deepseek-ai/DeepSeek-V4.1-Flash"])
def test_the_default_sm100_launch_keys_the_mxfp8_granularity_it_runs(model_id):
    """The audited default is TRT-LLM MXFP8 at group 32, not DeepGEMM at 128.

    The launch is the fact that changes the key: even DeepSeek-V4's group-128
    checkpoint request is eaten by the default backend at its own group 32.
    There is no FlashInfer provider yet, so this is a counted gap rather than
    a green measurement contract.
    """
    cells = [job for job in _moe_jobs(model_id, "B200")
             if job.get("signature") == "grouped_gemm"
             and "fp4" in str(job["dtype_env"].get("weight", ""))]
    assert {(job["dtype_env"].get("activation"),
             job["dtype_env"].get("activation_granularity"))
            for job in cells} == {("fp8_e4m3", "per-group:32")}


@pytest.mark.parametrize("model_id", BLACKWELL_UNRESOLVED)
def test_an_unresolved_blackwell_cell_stays_a_visible_gap(model_id):
    """No green assertion pretends the Blackwell question is settled.

    MiMo's vLLM activation is still unruled. The plan states the checkpoint's
    group-32 request, and the provider refuses that granularity rather than
    measuring DeepGEMM's group-128 launch silently.
    """
    jobs, _notes, unplaced = build_jobs_with_unplaced(
        model_id=model_id, batch_size=8, seq_len=1024, phase="decode",
        tp=8, framework="vllm", avg_output_tokens=512, gpu="B200")
    cells = [job for job in jobs
             if job.get("signature") == "grouped_gemm"
             and "fp4" in str(job["dtype_env"].get("weight", ""))]
    assert cells == []
    assert any(
        "precision launch selector gap" in record.get("detail", "")
        for record in unplaced)


@pytest.mark.parametrize("model_id,b200_activation", [
    ("openai/gpt-oss-20b", "bf16"),
    ("moonshotai/Kimi-K3", None),
])
def test_a_weight_only_mxfp4_moe_keeps_its_audited_card_boundary(
        model_id, b200_activation):
    """A checkpoint whose scheme declares no activation request stays W4A16
    where an audited launch exists.

    gpt-oss's distinct `mxfp4` scheme declares W4A16 and emits BF16 cells on
    both cards.  Kimi's compressed-tensors pack scheme also states no
    activation request, but its only declared pre-Blackwell launch is a
    deployment fact that is false on B200; that card boundary stays a visible
    selector gap rather than being repriced as a BF16 cell.
    """
    assert _moe_activations(model_id, "H100-SXM") == {"bf16"}
    assert _moe_activations(model_id, "B200") == (
        {"bf16"} if b200_activation is not None else set())
    if b200_activation is None:
        _jobs, _notes, unplaced = build_jobs_with_unplaced(
            model_id=model_id, batch_size=8, seq_len=1024, phase="decode",
            tp=8, framework="vllm", avg_output_tokens=512, gpu="B200")
        assert any(
            "precision launch selector gap: "
            "missing_deployment_fact:mxfp4_pack_quantized_weight_only"
            in record.get("detail", "")
            for record in unplaced)


# -- the provider end of the same statement -------------------------------

_MOE_DIMS = {"n_experts": 8, "n_tokens": 64, "s_expert_rows": 8,
             "d_reduce": 256, "d_out": 512}


def _moe_job(activation: str, granularity: str | None = None, *,
             explicit_deepgemm: bool = False) -> dict:
    # An expert GEMM writes its product wide whichever width it read — each
    # signature says so on its own `output=`; only the ACTIVATION slot varies
    # between these cells, which is the whole point of the pair.
    env = {"weight": "mxfp4", "activation": activation, "output": "bf16",
           "metadata": "ue8m0", "weight_granularity": "per-group:32"}
    if granularity is not None:
        env["activation_granularity"] = granularity
    params = {"moe_token_distribution": "balanced"}
    if explicit_deepgemm:
        params["moe_backend"] = "deep_gemm"
    return {"signature": "grouped_gemm", "op": "grouped_gemm",
            "phase": "prefill", "tp": 1, "dims": dict(_MOE_DIMS),
            "dtype_env": env,
            "kernel_params": params,
            "physical_key": ["grouped_gemm",
                             json.dumps(_MOE_DIMS, sort_keys=True),
                             json.dumps(env, sort_keys=True), "prefill"]}


@pytest.fixture
def h100(monkeypatch):
    monkeypatch.setenv("KERNEL_BENCH_WORK_DIR", "/nonexistent")
    profile = find_box_profile("h100-sxm-cu128", ENVS_DIR)
    return BoxEnvs.of(profile, ENVS_DIR, root=ROOT)


def test_the_weight_only_mxfp4_cell_has_a_kernel_on_an_h100(h100, tmp_path):
    """vLLM's W4A16 Marlin expert GEMM (`vllm_marlin_mxfp4_grouped_gemm`,
    no `min_sm`) is what serves these cells, and it has been registered all
    along — it could never be reached while they were keyed FP8."""
    plan = tmp_path / "plan.jsonl"
    _queued, gaps = write_plan([_moe_job("bf16")], str(plan), h100)
    assert gaps == []
    (job,) = read_jobs_jsonl(str(plan))
    assert "vllm_marlin_mxfp4_grouped_gemm" in job["candidates"]


def test_the_blackwell_fp8_fp4_cell_is_still_refused_on_an_h100(h100,
                                                               tmp_path):
    """A synthetic group-128 cell cannot bypass the missing backend selector.

    On this H100 the card is also true, but the provider is disabled even on
    Blackwell until a request can say `moe_backend=deep_gemm` (or prove the
    FlashInfer fallback). The cell stays in the coverage denominator.
    """
    plan = tmp_path / "plan.jsonl"
    _queued, gaps = write_plan(
        [_moe_job("fp8_e4m3", "per-group:128", explicit_deepgemm=True)],
        str(plan), h100)
    assert read_jobs_jsonl(str(plan)) == []
    (gap,) = read_jobs_jsonl(gaps_path(str(plan)))
    assert gap["no_provider"]["deepgemm_fp8_fp4_m_grouped_contiguous"] == (
        "sm_90 is below sm_100")
    # A gap the box cannot run is NOT a keying error: some kernel reads it.
    assert "unreadable_key" not in gap
    assert not any("KEYING ERROR" in line
                   for line in gaps_by_signature(gaps))

    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers

    job = _moe_job("fp8_e4m3", "per-group:128")
    invocation = Invocation(
        signature=job["signature"], op=job["op"], dims=job["dims"],
        dtype_env=job["dtype_env"], phase=job["phase"], tp=job["tp"],
        kernel_params=job["kernel_params"])
    provider = next(p for p in all_providers()
                    if p.kernel_impl == "deepgemm_fp8_fp4_m_grouped_contiguous")
    assert not provider.serves.accepts(invocation)
    assert "requires_params:moe_backend" in next(
        provider.serves.refusals(invocation))
    assert "FlashInfer TRT-LLM" in provider.disabled_reason

    selected = _moe_job("fp8_e4m3", "per-group:128", explicit_deepgemm=True)
    selector_invocation = Invocation(
        signature=selected["signature"], op=selected["op"],
        dims=selected["dims"], dtype_env=selected["dtype_env"],
        phase=selected["phase"], tp=selected["tp"],
        kernel_params=selected["kernel_params"])
    # Even the explicit selector does not bypass the disabled provider: no
    # production request carries that fact yet, so this remains a gap.
    assert provider.serves.accepts(selector_invocation)
    assert not provider.accepts(selector_invocation, object())
    assert next(
        provider.refusals(selector_invocation, object())) == "disabled"


def test_a_cell_no_kernel_anywhere_reads_is_named_a_keying_error(h100,
                                                                tmp_path):
    """The other half: a key nothing in the corpus reads on any card is the
    cost model's error, not a missing card, and it says so instead of
    sitting in the gap list looking like hardware nobody owns."""
    plan = tmp_path / "plan.jsonl"
    _queued, gaps = write_plan([_moe_job("fp8_e5m2")], str(plan), h100)
    (gap,) = read_jobs_jsonl(gaps_path(str(plan)))
    assert "activation=fp8_e5m2" in gap["unreadable_key"]
    (_head, keying) = gaps_by_signature(gaps)
    assert keying.strip().startswith("KEYING ERROR: 1 cell(s)")
