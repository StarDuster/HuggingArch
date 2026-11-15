import json,statistics,datetime,sys,dataclasses
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path.insert(0,str(W/'calibration-worktree'))
from kernel_bench.env import detect_env,_probe_version
from kernel_bench.invocation import Invocation
from kernel_bench.providers import Provider,LatencyStats
from kernel_bench.emit import build_row,rows_to_csv
from backend.inference.benchmarks import validate_text
from backend.inference.op_calibration import CalibrationRegistry,merge_validated_rows,BenchmarkKey
jobs=[json.loads(l) for l in (R.parent/'jobs.jsonl').read_text().splitlines()]
summary=json.loads((R/'summary.json').read_text());all_ranks=[]
for rank in range(8):all_ranks+=json.loads((R/f'rank{rank}.json').read_text())
module='sglang.kernels.ops.communication.all_reduce_residual'
env=detect_env();env.libs[module]=_probe_version(module);env.library='sglang'
(R/'export-environment.json').write_text(json.dumps(dataclasses.asdict(env),indent=2)+'\n')
p=Provider(library='sglang',signature='all_reduce',kernel_impl='sglang_multicast_push_res_tp8_graph',import_name=module,applies_to=lambda inv,env:True,build=lambda *args:None)
rows=[]
for s in summary:
 n=s['n_comm_elems'];matching=[]
 for j in jobs:
  if j['signature']=='all_reduce' and json.loads(j['physical_key'][1])['n_comm_elems']==n:matching.append(j)
 assert len(matching)==1,(n,len(matching))
 inv=Invocation.from_job(matching[0]);measurements=[r for r in all_ranks if r['n_comm_elems']==n]
 assert len(measurements)==8 and all(m['cross_check']=='passed' for m in measurements)
 stats=LatencyStats(p50_us=s['max_rank_p50_us'],p95_us=max(m['p95_us'] for m in measurements),mean_us=max(statistics.mean(m['samples_us']) for m in measurements),samples=30,timing_method='warm_l2')
 row=build_row(inv,p,stats,env,date=datetime.date.today().isoformat(),contributor='StarDuster TP8 multicast, Gloo FP32 exact integer oracle, seeds17/43',cross_check='passed',rel_err=0.0,cosine=1.0,xcheck_typical=0.0,xcheck_worst=0.0,run_id='b300-tp8-multicast-20261115',box='B300 TP8, max rank median, graph128',kernel_symbol='')
 # This is a standalone distributed driver, not the single-device launch tracer.
 row['launch_signature']=f'sglang.kernels.ops.communication.all_reduce_residual.all_reduce_push_res(8, bf16[{n}], None); CUDA graph 128 calls; rank-max median'
 rows.append(row)
text=rows_to_csv(rows);_,validated=validate_text(text)
merge_validated_rows(CalibrationRegistry(),validated,source='distributed TP8 communication')
(R/'calibration_rows.json').write_text(json.dumps(validated,indent=2)+'\n');(R/'measurements.csv').write_text(text)
keys={BenchmarkKey.from_csv_row(row).to_physical() for row in validated};assert len(keys)==4
base=json.loads((R.parent/'library_kernel_calibration_rows.json').read_text());existing={BenchmarkKey.from_csv_row(row).to_physical() for row in base};assert not keys&existing
combined=base+validated;merge_validated_rows(CalibrationRegistry(),combined,source='library kernels plus TP8')
(R.parent/'library_plus_tp8_calibration_rows.json').write_text(json.dumps(combined,indent=2)+'\n')
print('Exported 4 exact physical keys. Combined overlay:',len(combined),'rows. Timing regime is warm_l2 CUDA graph; this is not fused AR+norm timing.')
