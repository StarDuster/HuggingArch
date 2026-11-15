import json,sys
from pathlib import Path
from kernel_bench.plan import build_jobs_with_unplaced
from backend.inference.parallel_config import ParallelConfig

R=Path(__file__).resolve().parent
whole='--whole' in sys.argv
signature='kimi_k31_trtllm_moe_span' if whole else 'kimi_k31_trtllm_latent_span'
stem='whole-moe' if whole else 'runtime-span'
if not whole:
    from dataclasses import replace
    from backend.inference.framework_bindings import FRAMEWORK_BINDINGS
    binding=FRAMEWORK_BINDINGS['sglang']
    FRAMEWORK_BINDINGS['sglang']=replace(binding,op_implementations=tuple(
        entry for entry in binding.op_implementations if entry[0]!='kimi_k31_trtllm_moe_span'))
rows=[];all_rows=[];seen=set()
for batch in [1,8]:
    jobs,notes,unplaced=build_jobs_with_unplaced(model_id='moonshotai/Kimi-K3.1-Pro',batch_size=batch,
        seq_len=16384,avg_output_tokens=512,phase='decode',framework='sglang',
        parallel=ParallelConfig(tp_attention=8,tp_dense_ffn=8,tp_moe_expert=4,ep_moe=2,dcp=8),
        gpu='B300',kv_block_sizes=[64])
    selected=[j for j in jobs if j['signature']==signature]
    print('batch',batch,'span jobs',len(selected),flush=True)
    print('unplaced span',[u for u in unplaced if 'trtllm_' in str(u)][:3],flush=True)
    assert sum(j['count_per_step'] for j in selected)==92
    all_rows.extend(selected)
    for j in selected:
        key=json.dumps(j['physical_key'],sort_keys=True)
        if key not in seen:
            seen.add(key);rows.append(j)
    (R/f'runtime-plan-{batch}-unplaced.json').write_text(json.dumps(unplaced,indent=2,default=str)+'\n')
    for j in rows:print(j['physical_key'],j['dims'],flush=True)
(R/(stem+'-jobs-all.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
(R/(stem+'-jobs.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
assert len(rows)==2, 'Native identity projection missing: do not fabricate calibration keys'
