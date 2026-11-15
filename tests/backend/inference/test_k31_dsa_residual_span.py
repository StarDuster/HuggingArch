"""DSA/DCP and residual spans preserve real dataflow and deployment gates."""
from types import SimpleNamespace
import pytest
from backend.inference.framework_bindings import DispatchCondition,get_binding,ServingDeployment
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.inference.gpu_specs import get_gpu
from tests.snapshots import resolve_snapshot

@pytest.fixture(scope='module')
def full_analysis():
    binding=get_binding('sglang').deployed(ServingDeployment.at(
        gpu_sku='B300',activation_dtype='bf16',tensor_parallel=8,kv_block_size=64))
    return analyze(resolve_snapshot('moonshotai/Kimi-K3.1-Pro'),
        parallel=ParallelConfig(tp_attention=8,tp_dense_ffn=8,tp_moe_expert=4,ep_moe=2,dcp=8),
        framework_binding=binding,phase='decode',batch_size=1,
        avg_input_tokens=2048,avg_output_tokens=512,gpu=get_gpu('B300'),breakdown=False)

def test_dsa_span_owns_preparation_cache_and_both_dcp_protocols(full_analysis):
    result=full_analysis
    rows=[r for r in result.per_op_timing if r['op']=='kimi_k31_dsa_span']
    assert sum(r['count_per_step'] for r in rows)==24
    for row in rows:
        assert row['shape_in']==[1,1,7168]
        assert row['shape_out']==[1,1,1536]
        assert row['kv_read_bytes_per_rank']>0
        assert row['comm_world']==8
        meta=result.laws.comm_meta.values[row['uid']]
        assert {m['op'] for m in meta['span_collectives']}=={'all_gather','all_to_all'}
        assert len(meta['span_runtime_stages'])==3
        assert meta['span_heterogeneous_payloads']
        node=result.program.by_uid(row['uid'])
        assert {'kv_cache_out','indexer_cache_out'} <= set(node.output_ports.values())
    assert not any(r['op'] in ('indexer_logits','flash_dsa','dsa_topk_mask','runtime_stage')
                   for r in result.per_op_timing)

def test_all_outer_residual_adds_are_owned_once(full_analysis):
    rows=full_analysis.per_op_timing
    counts={op:sum(r['count_per_step'] for r in rows if r['op']==op) for op in (
        'kimi_k31_attention_output_span','kimi_k31_attention_output_residual_span',
        'kimi_k31_trtllm_moe_span','kimi_k31_trtllm_moe_residual_span')}
    assert counts==dict(kimi_k31_attention_output_span=8,
        kimi_k31_attention_output_residual_span=85,kimi_k31_trtllm_moe_span=7,
        kimi_k31_trtllm_moe_residual_span=85)
    assert not any(r['op']=='add' for r in rows)
    for row in rows:
        if row['op']=='kimi_k31_attention_output_residual_span':
            assert row['shape_in']==[[1,1,1536],[1,1,7168]]

def test_direct_dcp_mixed_payloads_preserve_bytes_floors_and_annotations():
    from backend.inference.serving import _comm_origins
    from backend.arch.layout.comm_volume import ring_factor
    def event(op,width,n,protocol):
        return dict(op=op,comm_world=8,comm_base_stride=1,payload_dtype_bytes=width,
            tensor_bytes=width*n,tensor_bytes_extent=width*n,
            payload_extents=(n,),collective_in_extents=(n,),collective_out_extents=(n,),
            sparse_dcp_protocol=protocol)
    first=event('all_gather',8,3072,'candidate_multicast')
    second=event('all_to_all',2,13056,'projected_push_or_fi_a2a')
    m=_comm_origins(({},dict(sparse_dcp_local_work='annotation'),first,
        dict(op='runtime_stage',runtime_read_bytes=4),second))
    assert m['tensor_bytes']==24576+26112
    assert m['payload_dtype_bytes']==1
    assert m['payload_extents']==(24576+26112,)
    assert m['span_collectives']==(first,second)
    assert len(m['span_annotations'])==len(m['span_runtime_stages'])==1
    assert ring_factor('all_gather',8)==ring_factor('all_to_all',8)
    # Only these explicitly lowered protocols share one rate; differing
    # group widths or an ordinary collective continue to be refused.
    with pytest.raises(ValueError,match='heterogeneous'):
        _comm_origins((first,{**second,'comm_world':4}))
    with pytest.raises(ValueError,match='heterogeneous'):
        _comm_origins((first,{**second,'sparse_dcp_protocol':'nccl'}))

def test_rank_group_gate_does_not_claim_that_dcp_shards_query_heads():
    condition=DispatchCondition.parse({'world_sizes':{'dcp':[8]}},'test')
    assert condition.as_dict()=={'world_sizes':{'dcp':[8]}}
    program=SimpleNamespace(architectures=())
    for degree in (1,4,8,16):
        analysis=SimpleNamespace(rule=SimpleNamespace(world_degree=lambda world:degree))
        assert condition.graph.admits_program(program,analysis)==(degree==8)

@pytest.mark.parametrize('value',[{'bogus':[8]},{'dcp':[]},{'dcp':[True]},{'dcp':[0]},['dcp'],[],None])
def test_invalid_rank_group_gates_are_rejected(value):
    with pytest.raises(ValueError,match='world_sizes'):
        DispatchCondition.parse({'world_sizes':value},'test')

@pytest.mark.parametrize('flags',[{'enable_fp4_indexer':True},
    {'dcp_replicate_q_proj':False},{'SGLANG_ENABLE_DSA_DCP_FUSED_TOPK':True},
    {'SGLANG_ENABLE_DCP_PUSH_COMBINE':False}])
def test_recipe_variants_do_not_reuse_the_dsa_span(flags):
    b=get_binding('sglang').deployed(ServingDeployment.at(gpu_sku='B300',
        activation_dtype='bf16',tensor_parallel=8,kv_block_size=64,server_flags=flags))
    assert 'kimi_k31_dsa_span' not in [name for name,*_ in b.dispatched()]

def test_dsa_calibration_declines_other_owner_distributions(full_analysis):
    from backend.inference.calibration_overlay import (
        WorkloadShape,build_deployment_context,point_keys)
    row=next(r for r in full_analysis.per_op_timing if r['op']=='kimi_k31_dsa_span')
    deployment=build_deployment_context(framework='sglang',framework_binding=None,
        parallel=ParallelConfig(tp_attention=8,dcp=8),gpu_sku='B300',
        phase='decode',kv_block_size=64)
    def key_at(counts):
        moved={**row,'point_law':{**row['point_law'],'point':counts}}
        return point_keys(full_analysis,[moved],WorkloadShape(1,2048),deployment,
                          model_id='moonshotai/Kimi-K3.1-Pro',spec_hash='test')[0]
    counts=row['point_law']['point']
    assert not key_at(counts).declined
    for delta in ({'rho_dcp':1.25},{'dcp_failed_owners':1}):
        point=key_at({**counts,**delta})
        assert point.declined
        assert point.placement_reason=='declined'
    assert key_at({k:v for k,v in counts.items() if k!='rho_dcp'}).declined

def test_native_plan_retains_dsa_without_a_single_process_provider(full_analysis,monkeypatch):
    from backend.inference import pipeline
    from kernel_bench.plan import build_jobs
    from kernel_bench.providers import served_kv_block_sizes
    assert not served_kv_block_sizes('kimi_k31_dsa_span')
    monkeypatch.setattr(pipeline,'analyze',lambda *args,**kwargs:full_analysis)
    jobs,_notes=build_jobs(model_id='moonshotai/Kimi-K3.1-Pro',batch_size=1,
        seq_len=2048,avg_output_tokens=512,phase='decode',gpu='B300',
        kv_block_sizes=[64],resolved=resolve_snapshot('moonshotai/Kimi-K3.1-Pro'),
        parallel=ParallelConfig(tp_attention=8,tp_dense_ffn=8,tp_moe_expert=4,ep_moe=2,dcp=8))
    dsa=[j for j in jobs if j['signature']=='kimi_k31_dsa_span']
    assert sum(j['count_per_step'] for j in dsa)==24
    assert all(j['dims']['kv_block_size']==64 for j in dsa)
