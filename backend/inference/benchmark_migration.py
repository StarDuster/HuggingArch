"""Bringing a benchmark CSV forward to the current schema: the fill each
added column gets on a row that predates it, and the rewrite.

One engine, two callers. `tools/migrate_schema.py` runs it over files an
operator names; `kernel_bench.emit.IncrementalCsvWriter` runs it on a results file it is
about to append to, so a box whose CSVs predate a schema bump keeps writing
into them instead of refusing the whole sweep (the v16 deploy: every model
ended rc=1 on "header does not match schema" until the files were migrated
by hand). A schema bump therefore needs exactly one thing beyond the column
itself: its entry in `FILL` — which `test_benchmarks` holds every optional
column to.

What each fill means, and why none of them is a guess, is the history in
`tools/migrate_schema.py`'s docstring and on each entry below.
"""
from __future__ import annotations

import csv as _csv
import io
import json

from backend.inference import benchmarks as _benchmarks
from backend.inference.benchmark_identity import declares_routed_token_total
from backend.inference.op_signatures import op_signature

# Column -> what a row that predates it truthfully says. `cross_check` has a
# vocabulary word for "no record"; the two numbers do not, and empty is that
# word for them. Adding a column here is all a future migration needs.
FILL = {
    "cross_check": "unknown",
    "cross_check_rel_err": "",
    "cross_check_cosine": "",
    # Empty is the truthful answer and the only one available: the symbol is
    # an observation of a running GPU, so no amount of reading a finished CSV
    # can recover it. A placeholder here would be a claim about a kernel
    # nobody watched.
    "kernel_symbol": "",
    # v7 provenance: the sweep's run_id, the box profile's name, and the
    # routing load factor are all facts of the moment of measurement. Empty
    # is the truthful answer for every row that predates the column.
    "run_id": "",
    "box": "",
    "measured_load_factor": "",
    # v8: whether the row is in a MoE expert block. A pre-v8 CSV predates the
    # declared dispatch zone; empty is the truthful answer (the block rollup
    # is simply unavailable for it until it is re-emitted).
    "moe_block_member": "",
    # v9: the seed the cell's inputs were drawn from. A pre-v9 row drew from
    # the process-global RNG, so no seed exists to recover; empty is the
    # truthful answer.
    "input_seed": "",
    # v10: the precision the kernel COMPUTED at where it is not the cell's
    # (a provider's declared fallback). Empty is what the column means for a
    # provider that declares none, so it is also what a row predating the
    # column truthfully says.
    "measured_dtype_env_json": "",
    # v11: the call that was timed, with its operands. An observation of a
    # running launch, so a finished CSV cannot be asked for it — empty is the
    # truthful answer and the column's own word for "not observed".
    "launch_signature": "",
    # v12: whether the row is a measurement or a refusal (the card could not
    # hold the cell). `measured` rather than empty, and this one IS
    # recoverable: before the column existed the harness wrote a row only
    # when it had timed one, so every row that predates it is a measurement.
    "outcome": "measured",
    # v13: the two numbers the cross-check verdict was made on. A finished
    # CSV cannot be asked for them — they exist only while the oracle's
    # output is still in memory — so empty is the truthful answer and the
    # column's own word for "no comparison was made". (Missed when v13
    # landed, which left this tool refusing every v12 CSV it was handed.)
    "xcheck_typical": "",
    "xcheck_worst": "",
    # v14: which driver ran the row. EMPTY, and this is the one fill where
    # the temptation to do better must be refused: the box's driver today is
    # not the driver that machine ran months ago, and `box` names a profile
    # two machines share and that gets edited. A presumed driver written here
    # would manufacture exactly the fact the pair exists to check — and would
    # read as "native", which is the claim nobody verified.
    "kmd_version": "",
    "umd_version": "",
    # v15: why a row that is not a measurement is one. Empty is the only
    # truthful answer for a row that predates it: every such row is a
    # measurement or an OOM, and neither owes a reason.
    "outcome_detail": "",
    # v16: which clock timed the row. Empty, and that is the column's own
    # word for "timed before the clock recorded itself" — the old memset
    # flush. Not recoverable from `date` alone (a sweep straddling the change
    # writes both), and a presumed value would claim the drain for rows that
    # did not have it.
    "timing_method": "",
}


# What the last `migrate_text` actually had to fill, so the closing note
# describes THIS migration. It used to print the v3 cross-check sentence
# unconditionally, which on a v4→v5 run claimed to have stamped a column it
# never touched.
FILLED: set = set()


def migrate_text(text: str) -> str:
    """Return the current-schema text, or the input unchanged if already there.

    Parsed and re-emitted with the shared writer so the result is byte-identical
    to what `emit_csv` would produce — the round-trip guarantee the schema
    doc promises has to survive a migration too.
    """
    rdr = _csv.reader(io.StringIO(text))
    rows = list(rdr)
    if not rows:
        return text
    header = rows[0]
    missing = [c for c in _benchmarks.ALL_COLUMNS if c not in header]
    unknown = [c for c in missing if c not in FILL]
    if unknown:
        # A column with no declared fill is one this tool does not know how to
        # answer truthfully. Refusing beats writing a plausible blank into a
        # column whose blank might mean something.
        raise ValueError(f"no declared fill for new column(s): {unknown}")
    FILLED.update(missing)
    body = [dict(zip(header, r, strict=False)) for r in rows[1:] if r]
    changed = bool(missing)
    for r in body:
        for column in missing:
            r[column] = FILL[column]
        row_changed = bool(missing)
        # Same question the validator asks, from the same declaration: which
        # kernels' cells carry a routed token total. A pair of op names here
        # was the third copy of that fact.
        if declares_routed_token_total(op_signature(str(r.get("op") or ""))):
            shape = json.loads(r["shape_json"])
            if "n_tokens" not in shape:
                try:
                    shape["n_tokens"] = int(shape["n_experts"]) * int(shape["s_query"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "cannot recover grouped-GEMM n_tokens without positive "
                        "n_experts and s_query") from exc
                r["shape_json"] = json.dumps(
                    shape, separators=(",", ":"), sort_keys=True)
                row_changed = True
        if row_changed:
            r["schema_version"] = str(_benchmarks.KNOWN_SCHEMA_VERSION)
        changed = changed or row_changed
    if not changed:
        return text
    return _benchmarks.emit_csv(list(_benchmarks.ALL_COLUMNS), body)
