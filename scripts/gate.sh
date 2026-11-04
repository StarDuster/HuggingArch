#!/usr/bin/env bash
# Fast gate for spec / naming / IR refactors — the loop you run per change.
#
# Covers the two things a naming or IR change can break: the checkpoint names
# every weight resolves to, and the layers that read them. The full suite is
# ~32 minutes and is a pre-push check, not an inner loop.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
fail=0

echo "── every weight's checkpoint name + element count, 39 snapshots ──"
$PY scripts/dump_tensor_names.py > /tmp/gate_names.txt 2>&1 || { echo "  dump 失败"; fail=1; }
if [ -f tests/golden/tensor_names.txt ]; then
  if diff -q tests/golden/tensor_names.txt /tmp/gate_names.txt > /dev/null; then
    echo "  ✓ $(wc -l < /tmp/gate_names.txt) 条全等"
  else
    echo "  ✗ 名字变了:"; diff tests/golden/tensor_names.txt /tmp/gate_names.txt | head -20; fail=1
  fi
else
  cp /tmp/gate_names.txt tests/golden/tensor_names.txt
  echo "  (基线不存在,已写入 tests/golden/tensor_names.txt)"
fi

echo "── the production storage surface, 39 snapshots ──"
# Pins what `Analysis.iter_checkpoint()` reports. Names, shapes, dtypes
# and bytes have one join; a second dump that recomputed them would
# drift. An intended correction still belongs in its own commit, with
# the evidence, before the surface moves.
$PY scripts/dump_storage_surface.py > /tmp/gate_storage.txt 2>&1 || { echo "  dump 失败"; fail=1; }
if [ -f tests/golden/storage_surface.txt ]; then
  if diff -q tests/golden/storage_surface.txt /tmp/gate_storage.txt > /dev/null; then
    echo "  ✓ $(wc -l < /tmp/gate_storage.txt) 条全等"
  else
    echo "  ✗ 存储面变了:"; diff tests/golden/storage_surface.txt /tmp/gate_storage.txt | head -20; fail=1
  fi
else
  cp /tmp/gate_storage.txt tests/golden/storage_surface.txt
  echo "  (基线不存在,已写入 tests/golden/storage_surface.txt)"
fi

echo "── targeted tests ──"
$PY -m pytest -q -p no:randomly --tb=line -rf \
  tests/backend/arch/test_spec_tree.py tests/backend/arch/test_spec_id_compose.py tests/backend/inference/test_fusion_registry.py \
  tests/backend/arch/test_op_names.py tests/backend/arch/validate/test_activation_check.py tests/backend/hf/test_forward_source_incomplete.py \
  tests/backend/arch/validate/test_validate_contract.py tests/backend/rendering/test_compute_graph_api.py \
  tests/backend/rendering/test_compute_graph_renderable.py \
  tests/backend/inference/test_sharded_shapes.py tests/backend/arch/eval/test_eval_checkpoint.py \
  || fail=1

exit $fail
