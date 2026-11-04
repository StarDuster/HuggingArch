"""The presentation layer: projections from ``eval.Analysis`` to the shapes
the pipeline, the API and the dashboard consume.

Projections, not a data model: every value here is assembled read-only from
analysis layers — no cost, shape, byte or multiplicity is recomputed outside
the derivations that own them. A projection exists because a consumer wants a
SHAPE (a tree of rows, a payload dict), and a wire shape needs exactly one
declared serializer; this package is that one place. Each module is one
consumer-facing surface:

    result    model and block reports used by validation and rendering,
              with the entries that analyze a spec and project it
    formulas  the dashboard's symbolic formula payload

The per-leaf HBM access rows are not here: they are a derived fact the
layout package's per-rank cost reads (`eval.access`), and keeping them in
this package put presentation on the composition root's import path.

The boundary ratchet (`tests/meta/test_eval_boundary.py`) pins what this package
may read — presentation may consume the analysis's public fields and the
shared IR substrate, and nothing else.
"""
from .result import (                                          # noqa: F401
    BlockReport, ModelReport, NodeReport,
    project_analysis, project_block, project_blocks, project_model,
)
from .formulas import display_formulas                  # noqa: F401
