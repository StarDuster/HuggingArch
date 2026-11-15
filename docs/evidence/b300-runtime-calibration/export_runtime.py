import json,sys
import datetime
import statistics
from pathlib import Path
from kernel_bench.env import detect_env,_probe_version
from kernel_bench.invocation import Invocation
from kernel_bench.providers import Provider,LatencyStats
from kernel_bench.emit import build_row,rows_to_csv
from backend.inference.benchmarks import validate_text
from backend.inference.op_calibration import CalibrationRegistry,merge_validated_rows

R=Path(__file__).resolve().parent
whole='--whole' in sys.argv
stem='whole-moe' if whole else 'runtime-span'
signature='kimi_k31_trtllm_moe_span' if whole else 'kimi_k31_trtllm_latent_span'
jobs=[json.loads(l) for l in (R/(stem+'-jobs.jsonl')).read_text().splitlines()]
env=detect_env();env.library='sglang'
module='sglang'
env.libs[module]=_probe_version(module)
provider=Provider(library='sglang',signature=signature,
    kernel_impl=('sglang_k31_tp8_ep2_whole_moe_two_streams_weights4_routes16' if whole else 'sglang_k31_tp8_ep2_prepared_trtllm_deferred_norm_graph_routes16'),
    import_name=module,applies_to=lambda i,e:True,build=lambda *args:None)
rank_rows=[x for rank in range(8) for x in json.loads((R/f'{"whole-moe" if whole else "latent-span"}-rank{rank}.json').read_text())]
rows=[];summary=[]
for batch in [1,8]:
    job=next(j for j in jobs if j['dims']['B']==batch)
    matching=[x for x in rank_rows if x['batch']==batch]
    assert len(matching)==8
    checks=[c for x in matching for c in x['checks']]
    assert len(checks)==16 and all(c['passed'] for c in checks)
    assert all(x['mesh']==dict(tp=8,ep=2,expert_tp=4) for x in matching)
    stat=LatencyStats(p50_us=max(x['p50_us'] for x in matching),
        p95_us=max(sorted(x['samples_us'])[28] for x in matching),
        mean_us=max(statistics.mean(x['samples_us']) for x in matching),
        samples=30,timing_method='warm_l2')
    row=build_row(Invocation.from_job(job),provider,stat,env,
        date=datetime.date.today().isoformat(),contributor='StarDuster actual K3 TP8/EP2 span, FP32+Gloo oracle, seeds17/43, rotating routes16',
        cross_check='passed',rel_err=max(c['rel'] for c in checks),cosine=min(c['cosine'] for c in checks),
        xcheck_typical=max(c['typical'] for c in checks),xcheck_worst=max(c['worst'] for c in checks),
        run_id='b300-k31-latent-execution-span',box='B300 eight ranks; maximum rank median; graph16 varied routes, no L2 drain')
    row['launch_signature']=('Whole K3 MoE _forward_fused source sequence, shared side stream, deferred finalize, both collectives and gemm_ag up/add, graph16 routes/4 dense weight sets' if whole else 'K3 fused_routing + _run_trtllm_prepared_routing_moe(do_finalize=False) + finalize_all_reduce_push_norm(world=8); no standalone intermediate quant or Norm charge')
    rows.append(row)
    summary.append(dict(batch=batch,max_rank_p50_us=stat.p50_us,numerical_checks=16,
        key=job['physical_key'],rank_p50_us=[x['p50_us'] for x in matching],
        excludes=[] if whole else ['dense front','shared-expert branch','latent up projection'],
        fixed_format='MXFP4 experts, MXFP8 GEMM inputs, BF16 norm weight, FP32 route logits; activation slot denotes internal expert MMA'))
csv=rows_to_csv(rows);_,validated=validate_text(csv)
merge_validated_rows(CalibrationRegistry(),validated,source='K31 distributed execution spans')
(R/(stem+'-calibration_rows.json')).write_text(json.dumps(validated,indent=2)+'\n')
(R/(stem+'-measurements.csv')).write_text(csv)
(R/(stem+'-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
base=json.loads((R.parent/'b300-calibration-followup/controlled-graph-warm-calibration_rows.json').read_text())
merge_validated_rows(CalibrationRegistry(),base+validated,source='Explicit runtime span overlay')
(R/(stem+'-overlay.json')).write_text(json.dumps(base+validated,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
