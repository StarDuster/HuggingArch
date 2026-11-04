"""The estimate and Kernel Bench planner read one walk-family registry."""
from __future__ import annotations

import inspect

from backend.inference import walk_registry


def test_capacity_and_plan_have_no_private_step_cells():
    """No entry point creates its own `StepCells` family."""
    from backend.inference import capacity
    from kernel_bench import plan

    for module, name in ((capacity, "_model_cells"), (plan, "_build_jobs_over_points")):
        source = inspect.getsource(getattr(module, name))
        assert "StepCells()" not in source, name
    assert "shared_walk_cells(" in inspect.getsource(capacity._model_cells)
    assert "shared_walk_cells(" in inspect.getsource(
        plan._build_jobs_over_points)


def test_a_walk_family_key_names_no_point_card_price_or_measurement():
    fields = tuple(walk_registry.WalkCellKey.__dataclass_fields__)
    assert fields == ("model_id", "version", "config", "parallel", "precision")
