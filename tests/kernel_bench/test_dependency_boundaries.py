"""Dependency direction for the chunk-launch fold invariant."""
from __future__ import annotations

import ast
from pathlib import Path

import kernel_bench.prefill_chunks as prefill_chunks
from backend.inference import chunk_terms, prefill_geometry


def test_kernel_bench_reexports_the_backend_owned_chunk_fold():
    assert prefill_chunks.ChunkLaunchTerm is chunk_terms.ChunkLaunchTerm
    assert prefill_chunks.merge_chunk_launch_terms is (
        chunk_terms.merge_chunk_launch_terms)
    assert prefill_chunks.chunk_term_count is chunk_terms.chunk_term_count


def test_kernel_bench_reexports_the_backend_owned_chunk_geometry():
    assert prefill_chunks.decompose_prefill is prefill_geometry.decompose_prefill
    assert prefill_chunks.prefill_chunk_map is (
        prefill_geometry.prefill_chunk_map)


def test_backend_chunk_policy_does_not_import_kernel_bench():
    source = Path(chunk_terms.__file__).read_text()
    source += Path(prefill_geometry.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert "kernel_bench" not in {
                name.name for name in node.names}, ast.dump(node)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "kernel_bench" and not (
                node.module or "").startswith("kernel_bench."), ast.dump(node)
