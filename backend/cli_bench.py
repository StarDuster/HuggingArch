"""Helpers for the `huggingarch bench` CLI subcommands.

Kept separate from `backend/cli.py` to avoid pulling the analyzer + arch
import chain into the legacy `huggingarch <model_id>` cold path: that path
never imports this module (`cli.py` imports it inside the bench subcommand),
which is where the laziness lives — so everything this module needs it
imports at the top, like any other module.
"""
from __future__ import annotations

import csv
import io
from typing import Any

from backend.arch.loader import SpecLoader
from backend.arch.sharding import PREFILL
from backend.hf.metadata import fetch_config
from backend.inference.benchmarks import ALL_COLUMNS, KNOWN_SCHEMA_VERSION
from backend.inference.calibration_overlay import (
    WorkloadShape,
    build_deployment_context,
    point_keys,
)
from backend.inference.framework_bindings import DEFAULT_FRAMEWORK, get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze


def emit_suggest_shapes(
    *,
    model_id: str,
    batch_size: int,
    seq_len: int,
    framework: str = DEFAULT_FRAMEWORK,
    precision: Any = None,
) -> str:
    """Run `pipeline.analyze` on `model_id` and emit a CSV template.

    One row per measurable op. Identity columns are filled where computable;
    the remaining identity columns and all measurement / provenance columns
    (except `schema_version` and `count_per_step`) are left as empty strings
    for the contributor to fill in before `bench validate` will pass.

    Rows whose identity shape does not project are OMITTED rather than
    templated — see `RowIdentity.of`. In practice that is the graph plumbing
    (`transpose`, `slice`, `view_split`, `concat`, …): ops nobody launches a
    kernel for, which used to arrive carrying the model's whole-step shape.

    `framework` selects the fusion rewrite — emitted CSV rows carry
    post-fusion op names, so the contributor's measurements must be taken
    with the same framework. Defaults to `DEFAULT_FRAMEWORK`; pass
    `--framework sglang` to template a sglang CSV instead. The contributor
    fills `framework_version` with the release they measured on.
    """
    config = fetch_config(model_id)
    if config is None:
        raise RuntimeError(f"could not fetch config for {model_id!r}.")
    loader = SpecLoader()
    resolved = loader.match(config, model_id=model_id)
    if resolved is None:
        raise RuntimeError(
            f"no spec matches {model_id!r}; run the spec generator first."
        )

    parallel = ParallelConfig()
    # Resolve the framework's binding so the rewrite runs
    # inside analyze. Without a binding, per_op_timing emits naive
    # pre-fusion op names and node_spec_ids; the serving route resolves
    # FUSED names, so contributor-authored CSVs would tuple-miss on
    # lookup.
    framework_binding = get_binding(framework)
    if framework_binding is None:
        raise RuntimeError(
            f"no framework binding shipped for {framework!r}. "
            f"Run `huggingarch fusion list-bindings` (or grep "
            f"backend/inference/framework_bindings/) to see what's available."
        )
    ar = analyze(
        resolved,
        parallel=parallel,
        framework_binding=framework_binding,
        batch_size=batch_size,
        avg_input_tokens=seq_len, avg_output_tokens=0,
        precision=precision,
        breakdown=True,
    )

    spec_hash = resolved.identity_hash()
    # Same owner as the plan and the serving-side lookup — the one key
    # derivation (`calibration_overlay.point_keys`) — so a row a contributor
    # measures from this template is keyed the way the request will ask for
    # it. The dtype environment is the LEAF's, per row: a model-wide default
    # templated an fp8 checkpoint's every row as bf16, and a contributor who
    # measured one in good faith added a row the estimate never asks for.
    #
    # Rows it refuses are LEFT OUT, and for the same reason: the uniform
    # `{B,S,dims}` template shape used to stand in, and that row named a
    # kernel the model never runs, under a key some other model's estimate
    # can hit. Naming the gap in `bench validate` is the fix, not filling it.
    # The paged-KV page in the key is the one this framework deploys with
    # (`build_deployment_context`), and absent where it declares none.
    points = point_keys(
        ar, ar.per_op_timing,
        WorkloadShape.of_walk(ar, seq_len=max(1, int(seq_len))),
        build_deployment_context(framework=framework,
                                 framework_binding=framework_binding,
                                 parallel=parallel, gpu_sku="", phase=PREFILL),
        model_id=model_id, spec_hash=spec_hash)

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(ALL_COLUMNS)
    for point in points:
        if point.key is None or point.declined:
            continue
        values = {c: "" for c in ALL_COLUMNS}
        values["model"] = model_id
        values["spec_hash"] = spec_hash
        values["op"] = point.op
        values["node_spec_id"] = point.aggregation_id
        values["shape_json"] = point.shape_json
        values["dtype_env_json"] = point.dtype_env_json
        values["count_per_step"] = str(point.count)
        values["schema_version"] = str(KNOWN_SCHEMA_VERSION)
        writer.writerow([values[c] for c in ALL_COLUMNS])
    return buf.getvalue()
