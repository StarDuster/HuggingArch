"""Serving layers for a test that prices rows off layers it built itself
(`phases.compute_per_op_timing`): the layers it names, every other one
absent, a book of its own."""
from __future__ import annotations

from dataclasses import fields
from typing import Any


def numbers_as_laws(**layers: Any) -> Any:
    """`pipeline.ServingLayers` holding `layers` (the layers after the
    rewrite: `cost`, `shapes`, `access`, ...) and nothing else. A fixture's
    layers are numbers, their own laws at any point."""
    from backend.arch.eval.layer import Layer
    from backend.inference.serving import ServingLayers
    from backend.inference.row_roofline import LawBook

    # A fixture's nodes rebind no names by a declared definition.
    given = {"book": LawBook(), "written": {}, "pre_written": {}, "collectives": {},
             "definitions": Layer(), "pre_definitions": Layer(), **layers}
    return ServingLayers(**{f.name: given.get(f.name)
                            for f in fields(ServingLayers)})
