"""The shape-sweep feasibility gate, unmocked.

`kernel_bench.sweep._ceilings_for_deployment` calls the capacity layer for
the deployment it gates. The sweep tests mock it, so a call that no longer type-checks would pass every
one of them while every real sweep died at the gate (kimi S4a/b review,
must-fix 1). This runs the real call once.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = PROJECT_ROOT / "backend" / "arch" / "snapshots" / "Qwen__Qwen3-8B.json"


def test_the_gate_runs_the_real_capacity_call():
    from backend.arch.eval.precision import AS_SHIPPED
    from backend.arch.loader import SpecLoader
    from kernel_bench.sweep import _ceilings_for_deployment, routine_parallel_configs
    config = json.loads(SNAPSHOT.read_text())["config"]
    # The sweep resolves the spec once and threads it — matching no longer
    # scores a config without a model_id, so the gate must be handed the spec
    # the same way `build_sweep_jobs` hands it (its single entry point).
    resolved = SpecLoader().match(config, model_id="Qwen/Qwen3-8B")
    _label, parallel = routine_parallel_configs(config, resolved=resolved)[0]
    by_seq = _ceilings_for_deployment(
        config=config, gpu="H100-SXM", parallel=parallel, seq_lens=[1024],
        avg_output_tokens=256,
        # The gate states its precision, as `build_sweep_jobs` does (as
        # shipped when the caller names none).
        precision=AS_SHIPPED, resolved=resolved)
    assert set(by_seq) == {1024}
    assert by_seq[1024].max_batch > 0
