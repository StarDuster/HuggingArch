"""P9 and P10: the checkpoint names and the storage surface, one block of
tab-separated rows per snapshot.

The invariant a naming or IR refactor must hold: whatever the spec graph is
called, the tensors it says are on disk do not move. `tensor_names.txt` holds
every weight's checkpoint name and element count; `storage_surface.txt` every
on-disk entry `Analysis.iter_checkpoint()` reports (name, shape, elements,
dtype, bytes, kind, copies), metadata siblings flattened so a line is one
safetensors entry. `scripts/dump_tensor_names.py` and
`dump_storage_surface.py` print these rows; the file keeps each snapshot's
block contiguous, in snapshot file-name order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from tools.corpus_pins.corpus import Corpus, CorpusModel
from tools.corpus_pins.pin import Doc, Pin, Row, Scope


def _analysis(model: CorpusModel) -> Any:
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.arch.loader import SpecLoader
    return Deployment(
        Program.of(SpecLoader().match(model.config, model_id=model.model_id))
    ).at(batch=1, input_tokens=8).checkpoint()


def tensor_name_rows(model: CorpusModel) -> List[str]:
    """The snapshot's rows, sorted; one `ERR` row when it does not analyze."""
    from backend.arch.eval.dim import value_of
    try:
        analysis = _analysis(model)
    except Exception as exc:  # reported as the row, not raised
        return [f"{model.stem}\tERR\t{type(exc).__name__}: {exc}"]
    # Through `tensors()`, not off the raw storage layer. The checkpoint name is
    # composed by TWO declarations — the op says `.weight` or a naked parameter,
    # the quant scheme says whether the bytes are that weight or a packed
    # encoding — and only the join has both. Reading `analysis.storage` alone
    # pinned eleven models' expert weights as `...w1.weight` while their
    # checkpoints hold `...w1.weight_packed`, and pinned it in the very file
    # whose stated invariant is "the tensors it says are on disk do not move".
    # One row per tensor, not per row of the join: a tensor stored at several
    # formats across its node's layers is several rows naming one tensor with
    # one element count, and `sorted` over a set of strings folds them.
    return sorted({
        f"{model.stem}\t{row.name}\t{int(value_of(row.elements))}"
        for row in analysis.tensors()
    })


def storage_surface_rows(model: CorpusModel) -> List[str]:
    """The snapshot's rows; one `!!` row when it does not analyze."""
    try:
        analysis = _analysis(model)
    except Exception as exc:  # reported as the row, not raised
        return [f"{model.stem}\t!!\t{type(exc).__name__}: {exc}"]
    rows = []
    for row in analysis.iter_checkpoint():
        kind = (
            "meta" if row.is_metadata
            else ("buffer" if row.is_buffer else "param")
        )
        rows.append((
            model.stem, row.name,
            "x".join(str(int(d)) for d in row.shape),
            str(int(row.elements)),
            row.storage_dtype or "",
            str(int(row.storage_bytes)),
            kind,
            str(int(row.count)),
        ))
    # (name, dtype) first, the whole row on a tie. One name recurs once per
    # layer group that holds it (dense and MoE layers, each with its own
    # count), and without the whole row the order among those entries is the
    # walk's order, which is not part of the surface.
    return ["\t".join(row) for row in sorted(rows, key=lambda r: (r[1], r[4], r))]


#: the second column of a row that reports a failed analysis
FAILED = ("ERR", "!!")


@dataclass(frozen=True)
class SurfacePin(Pin):
    rows: Callable[[CorpusModel], List[str]] = tensor_name_rows

    def parse(self, data: bytes) -> Doc:
        """`{stem: [row, ...]}` in file order. A snapshot's rows must be one
        contiguous block."""
        doc: Dict[str, List[str]] = {}
        last = None
        for line in data.decode("utf-8").splitlines():
            stem = line.split("\t", 1)[0]
            if stem != last and stem in doc:
                raise ValueError(f"{self.artifact}: {stem}'s rows are not contiguous")
            doc.setdefault(stem, []).append(line)
            last = stem
        return doc

    def render(self, doc: Doc) -> bytes:
        return "".join(line + "\n" for block in doc.values() for line in block).encode("utf-8")

    def merge(self, doc: Doc, row: Row) -> Doc:
        """Blocks in snapshot file-name order, the new block in its place."""
        merged = super().merge(doc, row)
        order = {stem: i for i, stem in enumerate(sorted(merged, key=lambda s: f"{s}.json"))}
        return dict(sorted(merged.items(), key=lambda item: order[item[0]]))

    def measure(self, model: CorpusModel) -> Row:
        return {model.stem: self.rows(model)}

    def failures(self, row: Row) -> List[str]:
        return [line for block in row.values() for line in block
                if line.split("\t")[1] in FAILED]

    def measure_all(self) -> Doc:
        return {model.stem: self.rows(model) for model in Corpus.models()}


_TEST = "tests/backend/arch/test_gate_surfaces_golden.py::"
TENSOR_NAMES = SurfacePin(
    id="P9",
    title="checkpoint names",
    scope=Scope.REQUIRED,
    artifact="tests/golden/tensor_names.txt",
    owner_tests=(_TEST + "test_every_snapshot_has_a_block",
                 _TEST + "test_block_equals_measurement"),
    rows=tensor_name_rows,
)
STORAGE_SURFACE = SurfacePin(
    id="P10",
    title="storage surface",
    scope=Scope.REQUIRED,
    artifact="tests/golden/storage_surface.txt",
    owner_tests=(_TEST + "test_every_snapshot_has_a_block",
                 _TEST + "test_block_equals_measurement"),
    rows=storage_surface_rows,
)
