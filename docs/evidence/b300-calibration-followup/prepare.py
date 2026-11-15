import json,sys,collections,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
REPO=R.parents[1]/'calibration-worktree';sys.path.insert(0,str(REPO))
from kernel_bench.plan import build_jobs_with_unplaced,write_plan
from backend.inference.parallel_config import ParallelConfig
jobs=[];records=[];notes=[];seen=set()
for batch in [1,8]:
 for seq in [2048,16384]:
  print('Planning',batch,seq,flush=True)
  js,ns,rs=build_jobs_with_unplaced(model_id='moonshotai/Kimi-K3.1-Pro',batch_size=batch,seq_len=seq,avg_output_tokens=512,phase='decode',framework='sglang',parallel=ParallelConfig(tp_attention=8,tp_dense_ffn=8,tp_moe_expert=4,ep_moe=2,dcp=8),gpu='B300',kv_block_sizes=[64])
  for j in js:
   key=json.dumps(j.get('physical_key',j),sort_keys=True)
   if key not in seen:seen.add(key);jobs.append(j)
  notes.extend(ns);records.extend(rs)
# Resolve existing measurements by the exact native PhysicalKey, not op names.
from backend.inference.op_calibration import BenchmarkKey,PhysicalKey,CalibrationRegistry,merge_validated_rows
from kernel_bench.emit import rows_to_csv
reports=R.parents[1]/'kimi-k31-reports'
revision=subprocess.check_output(['git','-C',str(reports),'rev-parse','origin/kimi-k31-reports'],text=True).strip()
raw=subprocess.check_output(['git','-C',str(reports),'show','origin/kimi-k31-reports:docs/evidence/kimi-k31-b300-calibration/calibration_rows.json'],text=True)
existing=json.loads(raw)
merge_validated_rows(CalibrationRegistry(),existing,source='kimi-k31-reports@'+revision)
existing_keys={BenchmarkKey.from_csv_row(row).to_physical() for row in existing}
new=[];reused=[]
for job in jobs:
 sig,shape,dtype,phase=job['physical_key']
 key=PhysicalKey(signature=sig,shape_json=shape,dtype_env_json=dtype,gpu='B300',phase=phase)
 (reused if key in existing_keys else new).append(job)
write_plan(jobs,str(R/'all-jobs.jsonl'),unplaced=records)
write_plan(new,str(R/'jobs.jsonl'),unplaced=records)
(R/'existing-calibration-rows.json').write_text(json.dumps(existing,indent=2)+'\n')
(R/'results').mkdir(exist_ok=True)
(R/'results/reused.csv').write_text(rows_to_csv(existing))
(R/'deduplication.json').write_text(json.dumps({'existing_branch':'StarDuster/HuggingArch:kimi-k31-reports','revision':revision,'existing_rows':len(existing),'planned_keys':len(jobs),'reused_plan_keys':len(reused),'new_plan_keys':len(new),'reused_jobs':reused},indent=2)+'\n')
print('Deduplicated:',len(jobs),'planned,',len(reused),'reused,',len(new),'new keys',flush=True)
(R/'plan-notes.json').write_text(json.dumps({'notes':notes,'job_count':len(jobs),'signatures':dict(collections.Counter(j['signature'] for j in jobs))},indent=2))
print('Jobs',len(jobs),'signatures',collections.Counter(j['signature'] for j in jobs),flush=True)
