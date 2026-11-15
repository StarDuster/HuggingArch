"""The distributed span keeps its math operands, communication and identity."""
import pytest
from dataclasses import replace
from backend.inference.framework_bindings import get_binding,ServingDeployment
from backend.inference.parallel_config import ParallelConfig
from backend.inference.gpu_specs import get_gpu
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot,snapshot_config


@pytest.fixture(scope='module')
def k31_span_analysis():
    model='moonshotai/Kimi-K3.1-Pro'
    binding=get_binding('sglang').deployed(ServingDeployment.at(
        gpu_sku='B300',activation_dtype='bf16',tensor_parallel=8,kv_block_size=64))
    # Keep the narrower span independently testable after the complete
    # two-stream block is bound at higher specificity.
    binding=replace(binding,op_implementations=tuple(
        entry for entry in binding.op_implementations if entry[0]!='kimi_k31_trtllm_moe_span'))
    return analyze(resolve_snapshot(model),parallel=ParallelConfig(
        tp_attention=8,tp_dense_ffn=8,tp_moe_expert=4,ep_moe=2,dcp=8),
        framework_binding=binding,phase='decode',batch_size=1,
        avg_input_tokens=2048,avg_output_tokens=512,gpu=get_gpu('B300'),breakdown=False)


def test_complete_moe_region_preserves_external_inputs_and_communication(k31_span_analysis):
    rows=[r for r in k31_span_analysis.per_op_timing if r['op']=='kimi_k31_trtllm_latent_span']
    assert sum(r['count_per_step'] for r in rows)==92
    for row in rows:
        assert row['shape_in']==[[1,1,896],[896],[1,1,3584]]
        assert row['shape_out']==[1,1,3584]
        assert row['comm_world']==8
        assert row['is_comm']
        assert row['routed'] is None


def test_owned_members_do_not_keep_separate_quant_norm_or_routing_rows(k31_span_analysis):
    rows=k31_span_analysis.per_op_timing
    assert not any(r['op']=='moe_fused_gate' for r in rows)
    assert not any(r['op']=='quantise_activation' and 'block_sparse_moe' in r['aggregation_id'] for r in rows)
    assert not any(r['op']=='rmsnorm' and 'routed_expert_norm' in r['aggregation_id'] for r in rows)


def test_capacity_supplies_actual_tp_to_dispatch(monkeypatch):
    from backend.inference import capacity
    seen=[]
    def capture(**kw):
        seen.append(kw['deployment'].tensor_parallel)
        raise RuntimeError('capacity deployment captured')
    monkeypatch.setattr(capacity,'_resolve_overlay_context',capture)
    with pytest.raises(RuntimeError,match='capacity deployment captured'):
        capacity.estimate_capacity(config=snapshot_config('Qwen/Qwen3-8B'),
            model_id='Qwen/Qwen3-8B',gpu_sku='H100-SXM',framework='sglang',
            num_gpus=2,parallel=ParallelConfig(tp_attention=2,tp_dense_ffn=2))
    assert seen==[2]


def test_same_group_collectives_keep_both_payloads_and_fixed_floors():
    import sympy as sp
    from backend.inference.row_roofline import build_row_exprs,COMM_S_PER_BYTE,COMM_LATENCY_S
    from backend.inference.serving import _comm_origins
    def comm(n):
        return dict(op='all_reduce',comm_world=8,comm_base_stride=1,
            payload_dtype_bytes=2,tensor_bytes=2*n,tensor_bytes_extent=2*n,
            payload_extents=(n,),collective_in_extents=(n,),collective_out_extents=(n,))
    merged=_comm_origins((comm(3584),comm(7168)))
    assert merged['tensor_bytes']==2*(3584+7168)
    assert len(merged['span_collectives'])==2
    law=build_row_exprs(count=1,flops=0,act_in=0,act_out=0,demand=0,
        weight_total=0,metadata_total=0,comm_payload=2*(3584+7168),comm_priced=True,
        comm_parts=((2*3584,1),(2*7168,1)),comm_dependency_rounds=2)
    assert sp.expand(law['t_comm'])==2*(3584+7168)*COMM_S_PER_BYTE+2*COMM_LATENCY_S


def test_collective_members_select_payload_rungs_individually():
    from backend.inference.row_roofline import build_row_exprs,COMM_S_PER_BYTE,COMM_LATENCY_S,rung_symbols
    law=build_row_exprs(count=1,flops=0,act_in=0,act_out=0,demand=0,
        weight_total=0,metadata_total=0,comm_payload=150,comm_priced=True,
        comm_parts=((60,1),(90,1)),comm_rungs=1,comm_dependency_rounds=2)
    bound,rate,fixed=rung_symbols(0)
    value=law['t_comm'].subs({bound:100,rate:1,fixed:3,COMM_S_PER_BYTE:10,COMM_LATENCY_S:20})
    # The aggregate exceeds100; each real message fits the fast rung.
    assert float(value)==156


def test_complete_moe_span_owns_both_streams_and_both_collectives():
    model='moonshotai/Kimi-K3.1-Pro'
    binding=get_binding('sglang').deployed(ServingDeployment.at(
        gpu_sku='B300',activation_dtype='bf16',tensor_parallel=8,kv_block_size=64))
    result=analyze(resolve_snapshot(model),parallel=ParallelConfig(
        tp_attention=8,tp_dense_ffn=8,tp_moe_expert=4,ep_moe=2,dcp=8),
        framework_binding=binding,phase='decode',batch_size=1,
        avg_input_tokens=2048,avg_output_tokens=512,gpu=get_gpu('B300'),breakdown=False)
    rows=[r for r in result.per_op_timing if r['op']=='kimi_k31_trtllm_moe_span']
    assert sum(r['count_per_step'] for r in rows)==92
    for row in rows:
        assert row['shape_in']==[[1,1,7168],[896]]
        assert row['shape_out']==[1,1,7168]
        assert row['routed'] is None
        assert row['comm_world']==8
        assert row['roofline_expr']['comm_dependency_rounds']==2
        assert len(result.laws.comm_meta.values[row['uid']]['span_collectives'])==2


def test_kda_projection_span_preserves_all_six_outputs(k31_span_analysis):
    result=k31_span_analysis
    rows=[r for r in result.per_op_timing if r['op']=='kimi_k31_kda_projection_span']
    assert sum(r['count_per_step'] for r in rows)==69
    nodes=[n for n in result.program.nodes() if n.op=='kimi_k31_kda_projection_span']
    for n in nodes:
        assert len(n.origins)==7
        assert set(n.output_ports.values())=={'','q_out','k_out','v_out','g_out','b_out'}
    assert not any(r['op']=='linear' and 'self_attn' in r['aggregation_id']
                   and any(r['aggregation_id'].endswith('.'+name) for name in (
                       'q_proj','k_proj','v_proj','g_proj','f_a_proj','f_b_proj','b_proj'))
                   and 'kda' in r['aggregation_id'] for r in result.per_op_timing)
    # The recurrence remains outside the measured projection contract.
    assert sum(r['count_per_step'] for r in result.per_op_timing
               if r['op']=='fused_kda_decode')==69
