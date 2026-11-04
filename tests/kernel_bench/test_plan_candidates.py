"""Every planned job has a declared provider on the box it is planned for.

A plan asks the declarations — each provider's `serves`, `min_sm` against
the SKU's compute capability, its version gate against what the box's env
files pin — of the env each library has on the box (`box.BoxEnvs`),
through the same gate a run takes (`Provider.declines`). A job no provider
would measure is written beside the plan as a gap, still counted by
`report`, and never handed to the GPU; `run-all` starts only the envs that
own a candidate of some job.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kernel_bench.box import BoxEnvs, find_box_profile, load_box_profile
from kernel_bench.plan import gaps_path, read_jobs_jsonl, write_plan
from kernel_bench.providers import all_providers, dependency_manifest

ENVS_DIR = str(ROOT / "kernel_bench" / "envs")


def judged_on(profile=None, *, manifest=None, envs_dir=ENVS_DIR, **box) -> str:
    """The premises `run-all` will find for a run on `profile` over
    `envs_dir` (with `--env` overrides etc. in `box`, and the test's own
    `manifest` when it fakes one) — what a test's hand-planned jobs carry
    so the run trusts their candidates instead of re-judging."""
    from kernel_bench.box import BoxEnvs
    from kernel_bench.providers import dependency_manifest
    return BoxEnvs.of(profile, str(envs_dir), **box).premises(
        manifest if manifest is not None else dependency_manifest())
_W = {"weight": "bf16", "activation": "bf16", "kv": "bf16", "output": "bf16",
      "metadata": "fp32"}


def _job(signature, op, phase, dims, dtype_env=_W, **extra):
    return {"signature": signature, "op": op, "phase": phase, "tp": 1,
            "dims": dims, "dtype_env": dict(dtype_env),
            "physical_key": [signature, json.dumps(dims, sort_keys=True),
                             json.dumps(dtype_env, sort_keys=True), phase],
            **extra}


SERVED = _job("gemm", "linear", "prefill",
              {"B": 1, "d_out": 5120, "d_reduce": 432, "s_query": 1024})
# A weight scale layout no registered kernel reads.
UNSERVED = _job("gemm", "linear", "prefill",
                {"B": 1, "d_out": 5120, "d_reduce": 432, "s_query": 1024},
                dtype_env={**_W, "weight": "fp8_e4m3",
                           "weight_granularity": "per-group:7"})
GROUPED = _job("grouped_gemm", "grouped_gemm", "prefill",
               {"n_experts": 8, "n_tokens": 64, "s_expert_rows": 8,
                "d_reduce": 256, "d_out": 512},
               kernel_params={"moe_token_distribution": "balanced"})


@pytest.fixture
def h100(monkeypatch):
    monkeypatch.setenv("KERNEL_BENCH_WORK_DIR", "/nonexistent")
    profile = find_box_profile("h100-sxm-cu128", ENVS_DIR)
    return profile, BoxEnvs.of(profile, ENVS_DIR, root=ROOT)


def test_a_job_no_declaration_accepts_is_a_gap_and_not_queued(
        h100, tmp_path, capsys):
    from kernel_bench.cli import main

    _profile, envs = h100
    plan = tmp_path / "plan.jsonl"
    write_plan([UNSERVED, SERVED], str(plan), envs)

    on_disk = read_jobs_jsonl(str(plan))
    assert [j["dtype_env"]["weight"] for j in on_disk] == ["bf16"]
    assert "torch_cublas_linear" in on_disk[0]["candidates"]
    assert on_disk[0]["candidates"]["torch_cublas_linear"] == "torch"
    # Beside the plan, under a name the queue does not read as one, and with
    # each targeting kernel's reason.
    assert not gaps_path(str(plan)).endswith(".jsonl")
    (gap,) = read_jobs_jsonl(gaps_path(str(plan)))
    assert gap["dtype_env"]["weight"] == "fp8_e4m3"
    assert gap["no_provider"]["torch_cublas_linear"] == (
        "weight_granularity:per-group:7 is not served (serves none); "
        "weight:fp8_e4m3 is not served (serves bf16, fp16, fp32)")
    # Still a cell: the report counts it.
    assert main(["report", "--jobs", str(plan), "--gpu", "H100-SXM"]) == 0
    out = capsys.readouterr().out
    assert "report: 2 cells" in out and "no provider 1" in out


def test_a_deepseek_v3_alignment_refusal_names_the_axis_and_modulus(h100):
    """The decode cell that made DeepGEMM's contiguous layout a gap.

    A missing activation-granularity declaration used to win the first-clause
    race and call the cell an activation-granularity gap. The launch really
    quantises at that group; once it says so, the contiguous kernel's own
    128-row alignment limit is the refusal that remains.
    """
    from kernel_bench.invocation import Invocation
    from kernel_bench.plan import build_jobs
    from kernel_bench.providers import candidates

    jobs, _dropped = build_jobs(
        avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3",
        batch_size=1, seq_len=1024, phase="decode")
    cell = next(j for j in jobs
                if j["signature"] == "grouped_gemm"
                and j["dims"].get("s_expert_rows") == 1
                and j["dims"].get("d_reduce") == 2048
                and j["dims"].get("d_out") == 7168)
    _profile, box = h100
    chosen, why = candidates(Invocation.from_job(cell), box.envs())
    assert "deepgemm_fp8_m_grouped_contiguous" not in {
        p.kernel_impl for p in chosen}
    assert why["deepgemm_fp8_m_grouped_contiguous"] == (
        "divisible:s_expert_rows=1 is not divisible by 128")
    # The masked launch pads that decode cell to its tile; contiguous cannot.
    assert "deepgemm_fp8_m_grouped_masked" in {
        p.kernel_impl for p in chosen}


def test_the_queue_takes_a_plans_gaps_with_it_and_not_as_a_plan(
        h100, tmp_path):
    from kernel_bench.queue import Workspace, hand_over, pending

    _profile, envs = h100
    space = Workspace(tmp_path / "box").prepare()
    staged = tmp_path / "staged.jsonl"
    write_plan([UNSERVED, SERVED], str(staged), envs)
    hand_over(space, "org/model", staged)
    assert pending(space) == ["org/model"]
    assert len(read_jobs_jsonl(gaps_path(str(space.plan_of("org/model"))))) == 1
    # A re-plan with no gaps leaves none behind.
    write_plan([SERVED], str(staged), envs)
    hand_over(space, "org/model", staged)
    assert not Path(gaps_path(str(space.plan_of("org/model")))).exists()


def test_run_all_starts_only_the_envs_a_grouped_gemm_plan_needs(
        h100, tmp_path, capsys):
    """An unannotated plan (written before plans carried candidates) is
    asked the same question by run-all, of the envs it can start."""
    from kernel_bench.cli import main
    from kernel_bench.plan import write_jobs_jsonl

    profile, _ = h100
    plan = tmp_path / "plan.jsonl"
    write_jobs_jsonl([GROUPED], str(plan))
    assert main(["run-all", "--jobs", str(plan), "--dry-run",
                 "--box", str(profile.path), "-o", str(tmp_path / "r")]) == 0
    out = capsys.readouterr().out
    started = {line[1:line.index("]")] for line in out.splitlines()
               if line.startswith("[") and "] PYTHONPATH=" in line}
    skipped = {line[1:line.index("]")] for line in out.splitlines()
               if line.startswith("[") and "] skip: no provider" in line}
    owners = {p.library for p in all_providers()
              if p.signature == "grouped_gemm"}
    assert started and started <= owners, started
    assert "flashinfer" in skipped and "fa3" in skipped
    assert started | skipped == set(dependency_manifest()), (started, skipped)


def test_a_version_gated_kernel_is_no_candidate_where_the_box_pins_past_it(
        tmp_path, monkeypatch):
    """sglang-kernel 0.4.5 retired the AOT `moe_fused_gate`
    (`version_specifier="<0.4.5"`). A box whose sglang env pins a later
    release declares an env that kernel cannot run in."""
    monkeypatch.setenv("KERNEL_BENCH_WORK_DIR", "/nonexistent")
    reqs = tmp_path / "sglang.txt"
    reqs.write_text(f"-r {ROOT / 'kernel_bench' / 'envs' / 'sglang.txt'}\n"
                    "sgl-kernel==0.4.9\n")
    box = tmp_path / "box.toml"
    box.write_text(f'name = "pinned"\n[env_requirements]\nsglang = "{reqs}"\n')
    envs = BoxEnvs.of(load_box_profile(box), ENVS_DIR, gpu="H100-SXM",
                      root=ROOT).envs()
    gate = next(p for p in all_providers()
                if p.kernel_impl == "sgl_moe_fused_gate")
    assert envs["sglang"].version_of("sgl_kernel") == "0.4.9"
    assert next(gate.unavailability(envs["sglang"]), None) == (
        "sgl_kernel 0.4.9 is outside <0.4.5")
    # And an env that pins one it still ships keeps it.
    reqs.write_text(f"-r {ROOT / 'kernel_bench' / 'envs' / 'sglang.txt'}\n"
                    "sgl-kernel==0.4.4\n")
    envs = BoxEnvs.of(load_box_profile(box), ENVS_DIR, gpu="H100-SXM",
                      root=ROOT).envs()
    assert next(gate.unavailability(envs["sglang"]), None) is None


GQA = _job("gqa_attention", "flash_attention", "prefill",
           {"B": 1, "d_head": 128, "kv_block_size": 16, "n_heads": 8,
            "n_kv_heads": 2, "s_kv": 1024, "s_query": 1024})


def test_a_box_is_judged_on_its_own_card_never_a_default(h100, capsys):
    """`sweep --box h100` used to judge on the RTX 4090 default: every sm_90
    kernel's cells written off as gaps."""
    from kernel_bench.box import BoxProfileError
    from kernel_bench.cli import main

    profile, box = h100
    assert box.gpu == "H100-SXM"
    with pytest.raises(BoxProfileError, match="not the card"):
        BoxEnvs.of(profile, ENVS_DIR, gpu="RTX-4090")
    assert main(["sweep", "any/model", "--box", str(profile.path),
                 "--gpu", "RTX-4090"]) == 2
    assert "not the card" in capsys.readouterr().err


def test_a_run_on_other_premises_judges_the_plan_again(tmp_path, monkeypatch,
                                                        capsys):
    """Planned where fa3 was not provisioned; run with `--env fa3=…`. The
    plan's answer rests on premises this run does not share, so run-all
    asks again — and starts fa3 — rather than trusting it."""
    from kernel_bench.cli import main

    monkeypatch.delenv("KERNEL_BENCH_ENV_FA3", raising=False)
    box_file = tmp_path / "box.toml"
    box_file.write_text('name = "h100-bare"\n')
    plan = tmp_path / "plan.jsonl"
    write_plan([GQA], str(plan),
               BoxEnvs.of(load_box_profile(box_file), ENVS_DIR, root=ROOT))
    assert "flash_attention_v3" not in read_jobs_jsonl(str(plan))[0]["candidates"]

    assert main(["run-all", "--jobs", str(plan), "--dry-run",
                 "--box", str(box_file), "-o", str(tmp_path / "r"),
                 "--env", "fa3=/opt/fa3/bin/python"]) == 0
    out = capsys.readouterr().out
    assert "re-judged" in out
    assert "[fa3] PYTHONPATH=" in out


def test_a_replan_without_a_box_leaves_no_earlier_gaps(h100, tmp_path):
    _profile, box = h100
    plan = tmp_path / "plan.jsonl"
    write_plan([UNSERVED, SERVED], str(plan), box)
    assert Path(gaps_path(str(plan))).exists()
    write_plan([UNSERVED, SERVED], str(plan))
    assert not Path(gaps_path(str(plan))).exists()
    assert len(read_jobs_jsonl(str(plan))) == 2


def test_a_needed_env_that_cannot_start_leaves_the_model_not_done(
        tmp_path, monkeypatch):
    """A plan that needs a source build this box has no interpreter for:
    those jobs were never tried, so the run is not done — unlike an env no
    job needs — and the queue neither marks it nor spins on it."""
    from kernel_bench import supervisor
    from kernel_bench.cli import main
    from kernel_bench.plan import write_jobs_jsonl
    from kernel_bench.queue import Workspace, run_once, serve

    monkeypatch.delenv("KERNEL_BENCH_ENV_FA3", raising=False)
    monkeypatch.setattr(supervisor, "run_command",
                        lambda *a, **k: supervisor.SupervisedResult(0, None, 0.1))
    plan = tmp_path / "plan.jsonl"
    write_jobs_jsonl([{**GQA, "candidates": {"flash_attention_v3": "fa3"},
                       "judged_on": judged_on()}], str(plan))
    run_all = ["run-all", "--jobs", str(plan), "-o", str(tmp_path / "r")]
    assert main(run_all) != 0

    space = Workspace(tmp_path / "box").prepare()
    write_jobs_jsonl(read_jobs_jsonl(str(plan)), str(space.plan_of("org/m")))
    calls = []

    def spawn(cmd):
        calls.append(cmd)
        return main(run_all)
    assert run_once(space, "org/m", spawn=spawn) != 0
    assert not space.done_marker("org/m").exists()
    serve(space, stop_when_idle=True, spawn=spawn)
    assert len(calls) == 2              # once more this pass, then set aside
    assert not space.done_marker("org/m").exists()


def test_an_all_gap_plan_is_judged_again_when_its_premises_change(
        tmp_path, monkeypatch, capsys):
    """Every job a gap, so the plan file is empty: the premises are only on
    the sidecar, and a run that ignored it went `.done` with fa3 now there."""
    from kernel_bench import providers as P
    from kernel_bench.cli import main

    monkeypatch.delenv("KERNEL_BENCH_ENV_FA3", raising=False)
    P.discover()
    fa3 = next(p for p in P._PROVIDERS if p.kernel_impl == "flash_attention_v3")
    monkeypatch.setattr(P, "_PROVIDERS", [fa3])
    box_file = tmp_path / "box.toml"
    box_file.write_text('name = "h100-bare"\n')
    plan = tmp_path / "plan.jsonl"
    queued, gaps = write_plan(
        [GQA], str(plan),
        BoxEnvs.of(load_box_profile(box_file), ENVS_DIR, gpu="H100-SXM",
                   root=ROOT))
    assert queued == [] and len(gaps) == 1

    assert main(["run-all", "--jobs", str(plan), "--dry-run",
                 "--box", str(box_file), "-o", str(tmp_path / "r"),
                 "--env", "fa3=/opt/fa3/bin/python"]) == 0
    out = capsys.readouterr().out
    assert "re-judged" in out and "[fa3] PYTHONPATH=" in out


def test_the_premises_are_the_pins_not_the_paths(tmp_path):
    """An edited pin is another premise; the same pins under another path
    are not."""
    def premises(pin, name):
        reqs = tmp_path / name
        reqs.write_text(f"-r {ROOT / 'kernel_bench' / 'envs' / 'sglang.txt'}\n"
                        f"sgl-kernel=={pin}\n")
        box = tmp_path / f"{name}.toml"
        box.write_text(f'name = "b"\n[env_requirements]\nsglang = "{reqs}"\n')
        return judged_on(load_box_profile(box), gpu="H100-SXM")
    assert premises("0.4.9", "a.txt") != premises("0.4.4", "a.txt")
    assert premises("0.4.9", "a.txt") == premises("0.4.9", "b.txt")


def test_the_queue_runs_the_env_files_its_plans_were_judged_on(tmp_path):
    from kernel_bench.queue import Workspace, run_all_command
    space = Workspace(tmp_path).prepare()
    cmd = run_all_command(space, "org/m", envs_dir="/custom/envs")
    assert cmd[cmd.index("--envs-dir") + 1] == "/custom/envs"


def test_a_box_whose_card_cannot_be_named_needs_an_explicit_gpu(
        tmp_path, capsys):
    from kernel_bench.cli import main
    box_file = tmp_path / "box.toml"
    box_file.write_text('name = "custom-box"\n')
    assert main(["sweep", "any/model", "--box", str(box_file)]) == 2
    assert "names no GPU SKU" in capsys.readouterr().err
