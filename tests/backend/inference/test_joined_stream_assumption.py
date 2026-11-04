"""A join's elements stay resident outside the activation peak, and an
estimate with a join says so."""
from __future__ import annotations

from backend.inference.assumptions import deployment_assumptions
from backend.inference.parallel_config import ParallelConfig


def test_a_join_holds_its_elements_outside_the_peak():
    """A join's elements stay resident from the layer that appends each to
    the join; the activation peak counts them only where they are joined,
    and an estimate with a join says so, at any mesh."""
    (note,) = [n for n in deployment_assumptions(
        ParallelConfig(), joined_streams=["aux"])
        if n["id"] == "joined_stream_residency"]
    assert note["effect"] == "optimistic" and "`aux`" in note["text"]
    assert "joined_stream_residency" not in [
        n["id"] for n in deployment_assumptions(ParallelConfig())]
