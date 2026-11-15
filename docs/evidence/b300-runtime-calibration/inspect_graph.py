import json
from pathlib import Path
from backend.arch.snapshots import load_snapshot
from backend.arch.loader import get_default_loader
from backend.inference.framework_bindings import get_binding, ServingDeployment
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.inference.gpu_specs import get_gpu, load_gpu_specs
from backend.inference.calibration_overlay import _law_to_json

R = Path(__file__).resolve().parent
model = 'moonshotai/Kimi-K3.1-Pro'
s = load_snapshot(model)
resolved = get_default_loader().match(s['config'], model_id=model)
load_gpu_specs()
binding = get_binding('sglang').deployed(ServingDeployment.at(gpu_sku='B300', activation_dtype='bf16', tensor_parallel=8, kv_block_size=64))
print('analyze', flush=True)
x = analyze(resolved, parallel=ParallelConfig(tp_attention=8,tp_dense_ffn=8,tp_moe_expert=4,ep_moe=2,dcp=8), framework_binding=binding, phase='decode', batch_size=1, avg_input_tokens=16384, avg_output_tokens=512, breakdown=False, gpu=get_gpu('B300'))
rows=[]
for n in x.pre_program.nodes():
    if 'block_dsa_moe[Sar=1]' in n.aggregation_id and 'block_sparse_moe' in n.aggregation_id and not n.children:
        rows.append(dict(uid=n.uid,op=n.op,path=n.aggregation_id,
            inputs=[dict(uid=r.src.uid if r.src else None,port=r.port,external=r.external_sid) for r in n.inputs_obj or ()],
            scope=x.pre_program.repeated_in(n.uid)))
(R/'moe-prefusion-graph.json').write_text(json.dumps(rows, indent=2, default=str)+'\n')
for r in rows:print(r,flush=True)
print('fusion matches',len(x.fusion_matches),flush=True)
span_rows=[r for r in x.per_op_timing if r['op']=='kimi_k31_trtllm_latent_span']
(R/'latent-span-model-rows.json').write_text(json.dumps(_law_to_json(span_rows),indent=2,default=str)+'\n')
print('latent spans:',len(span_rows),'calls',sum(r['count_per_step'] for r in span_rows),flush=True)
if span_rows:
    print('one span row', {k:span_rows[0].get(k) for k in ['shape_in','shape_out','weight_dtype','act_dtype','output_dtype','comm_world','routed','node_dims']},flush=True)
