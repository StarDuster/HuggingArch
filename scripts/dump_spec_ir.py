"""Every IR node the spec walk emits, over every snapshot.

The instrument for changes to how the loader BUILDS blocks. `iter_tree` is
downstream of every assembly decision — the block a layer resolves to, the
node path it lands at, the multiplicity it inherits — so a loader change that
alters any of them shows up here as a line diff, while the numbers a capacity
golden checks might still agree by coincidence.

Usage: `python scripts/dump_spec_ir.py > /tmp/before.txt`, change the loader,
dump again, diff.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.arch.loader import SpecLoader
from backend.arch.spec_tree import iter_tree

SNAPSHOTS = Path(__file__).resolve().parents[1] / "backend" / "arch" / "snapshots"


def _walk(nodes, model, scope, depth=0):
    for n in nodes:
        print("\t".join((
            model,
            scope,
            n.display_path,
            n.op,
            n.weight_name,
            str(n.dim),
            "x".join(str(d) for d in (n.shape or ())),
            str(n.multiplicity),
            str(n.active_multiplicity),
            n.role or "",
            n.template,
            n.module_name or "",
            "C" if n.is_component else "-",
        )))
        _walk(n.children, model, scope, depth + 1)


for path in sorted(glob.glob(str(SNAPSHOTS / "*.json"))):
    model = os.path.basename(path)[:-5]
    payload = json.loads(Path(path).read_text())
    try:
        resolved = SpecLoader().match(payload["config"], model_id=model.replace("__", "/", 1))
        tree = iter_tree(resolved)
    except Exception as exc:
        print(f"{model}\t!!\t{type(exc).__name__}: {exc}")
        continue
    _walk(tree.top_level, model, "top")
    for meta, roots in tree.blocks:
        # BlockMeta identity travels into `component_name`, node ids and
        # calibration keys — a renamed block is a changed system even when
        # every node under it is identical.
        print("\t".join((
            model, "meta",
            str(getattr(meta, "template_name", "")),
            str(getattr(meta, "stack_name", "")),
            str(getattr(meta, "count", "")),
            str(getattr(meta, "aux_kind", "")),
            str(getattr(meta, "group_name", "")),
            str(getattr(meta, "layer_indices", "")),
            str(getattr(meta, "is_singleton", "")),
            str(sorted((getattr(meta, "dim_overrides", None) or {}).items())),
        )))
        _walk(roots, model, f"block:{getattr(meta, 'template_name', '')}")
