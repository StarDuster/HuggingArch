import json,sys,collections
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parents[1]/'calibration-worktree'))
from backend.inference.benchmarks import validate_text
from backend.inference.op_calibration import CalibrationRegistry,merge_validated_rows
rows=[];failures=[]
for p in sorted((R/'results').glob('*.csv')):
 if p.name=='reused.csv':continue
 try:
  _,rs=validate_text(p.read_text());rows.extend(rs)
 except Exception as e:failures.append({'path':str(p),'error':str(e)})
passed=[r for r in rows if r.get('cross_check')=='passed' and r.get('latency_us') and float(r['latency_us'])>0 and r.get('outcome','') in ['', 'measured']]
merge_validated_rows(CalibrationRegistry(),passed,source='B300 native kernel_bench followup')
(R/'new_calibration_rows.json').write_text(json.dumps(passed,indent=2)+'\n')
existing=json.loads((R/'existing-calibration-rows.json').read_text())
combined=existing+passed
merge_validated_rows(CalibrationRegistry(),combined,source='reused plus new B300 measurements')
(R/'calibration_rows.json').write_text(json.dumps(combined,indent=2)+'\n')
summary={'reused_rows':len(existing),'combined_measured_rows':len(combined),'deduplication':json.loads((R/'deduplication.json').read_text()),'total_rows':len(rows),'passed_measured_rows':len(passed),'csv_validation_failures':failures,'outcomes':dict(collections.Counter(r.get('outcome','') or 'measured' for r in rows)),'measured_by_library':dict(collections.Counter(r['framework'] for r in passed)),'measured_by_op':dict(collections.Counter(r['op'] for r in passed)),'scope':'Kimi K3.1 target-only decode, TP8/DCP8/EP2, batch1/8, input2K/16K, output512, page64; native providers, not full AgentX or DFLASH model','coverage_limit':'Unsupported providers/signatures and distributed communication remain gaps. Single-GPU timings do not measure TP8 collectives or stream overlap. No full-measured latency claim.'}
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if not passed:raise SystemExit('No valid measured calibration rows')
