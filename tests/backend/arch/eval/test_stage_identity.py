"""A stage is joined to its own node, and a weight to its own stack.

A model-level op is where `spec.model` declares it: a group's op may share
its name with a model-root op (an MTP head's `lm_head`), and each runs as
its own stage. A block template bound by two stacks — an MTP layer built as
the backbone's layer — holds one set of weights per stack: the checkpoint
stores them under `model.layers.X.*` and `mtp.layers.X.*`.
"""
from __future__ import annotations

from pathlib import Path

from backend.arch.eval.program import OpStage, Program
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.persistent import physical_weight_leaves
from backend.arch.loader import SpecLoader
from tests.snapshots import resolve_snapshot, snapshot_config

ARCH = Path(__file__).resolve().parents[4] / "backend" / "arch"


def test_a_group_op_named_like_a_root_op_runs_its_own_node(tmp_path):
    text = (ARCH / "models" / "XiaomiMiMo__MiMo-7B-RL.yaml").read_text()
    text = text.replace('model_id: ["XiaomiMiMo/MiMo-7B-RL"]',
                        'model_id: ["test/same-name-head"]')
    assert "name: mtp_logits," in text
    text = text.replace("name: mtp_logits,", "name: lm_head,")
    models = tmp_path / "models"
    models.mkdir()
    (models / "test__same-name-head.yaml").write_text(text)
    loader = SpecLoader(overlay_model_dirs=[str(models)])
    resolved = loader.resolve(loader.get("test__same-name-head"),
                              snapshot_config("XiaomiMiMo__MiMo-7B-RL"))
    heads = [stage for stack in Program.of(resolved).stacks
             for stage in stack.stages
             if isinstance(stage, OpStage) and stage.node.name == "lm_head"]
    assert sorted(stage.stack for stage in heads) == ["", "mtp"]
    assert len({stage.node.uid for stage in heads}) == 2
    for stage in heads:
        assert stage.node.stack_name == (stage.stack or None)
    # The head's tie names the model-root `lm_head`, the earlier layer of
    # that name, not itself: it reads the weight the root head reads.
    root, head = sorted(heads, key=lambda stage: stage.stack)
    assert head.node.tied_to_uid == (root.node.tied_to_uid or root.node.uid)


def test_a_template_bound_by_two_stacks_holds_two_weights():
    resolved = resolve_snapshot("Qwen__Qwen3.8-27B")
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    def _node(leaf):
        return analysis.program.by_uid(leaf.node_uid)

    norms = {leaf.stack: leaf
             for leaf in physical_weight_leaves(analysis).values()
             if _node(leaf).template.startswith("qwen35_full_dense_block")
             and _node(leaf).display_path.endswith("layer.input_layernorm")}
    assert sorted(norms) == ["", "mtp"]
    assert norms[""].node_uid != norms["mtp"].node_uid
    assert (norms[""].global_instances, norms["mtp"].global_instances) == (16, 1)


def test_a_backbone_id_does_not_depend_on_other_stacks():
    """The backbone's ids are the same whether the MTP head is present."""
    def ids(n_mtp: int) -> set:
        config = dict(snapshot_config("Qwen__Qwen3.8-27B"))
        text = dict(config.get("text_config") or {})
        key = "mtp_num_hidden_layers"
        assert key in text
        text[key] = n_mtp
        config["text_config"] = text
        resolved = SpecLoader().match(config, model_id="Qwen/Qwen3.8-27B")
        program = Program.of(resolved)
        return {node.aggregation_id for node in program.nodes()
                if program.stack_of(node) == ""}
    assert ids(0) == ids(1)
