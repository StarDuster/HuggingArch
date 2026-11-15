import json,subprocess,statistics
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1];REPO=W/'calibration-worktree';PY=W/'HuggingArch/.venv313/bin/python'
results=R/'estimates';results.mkdir(exist_ok=True)
historical=json.loads((W/'benchmark/dcp-validation/results.json').read_text())['runs']
rows=[]
for seq in [2048,16384]:
 for stage,data in [('theory',None),('existing',R/'existing-calibration-rows.json'),('expanded',R/'calibration_rows.json')]:
  out=results/f'{seq}-{stage}.json'
  cmd=[str(PY),str(REPO/'scripts/estimate_kimi_k31_sglang.py'),'--dcp','8','--input-tokens',str(seq),'--output',str(out)]
  if data:cmd+=['--calibration-rows',str(data)]
  if stage=='expanded' or not out.exists():
   with (results/f'{seq}-{stage}.log').open('w') as log:subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
  summary=json.loads(out.with_suffix('.summary.json').read_text());phase=summary['phases']['decode'];measured=statistics.mean(x['mean_tpot_ms'] for x in historical if x['input']==seq)
  predicted=phase['t_partial_s'] if data else phase['t_sol_s']
  rows.append({'input_tokens':seq,'stage':stage,'predicted_ms':1000*predicted,'historical_tpot_ms':measured,'error_pct':100*(1000*predicted/measured-1),'t_sol_ms':1000*phase['t_sol_s'],'t_partial_ms':None if phase.get('t_partial_s') is None else 1000*phase['t_partial_s'],'t_measured_s':phase.get('t_measured_s'),'partial_measured_share':phase.get('partial_measured_share'),'measured_rows':phase.get('measured_rows'),'total_rows':phase.get('total_rows')})
(R/'estimate-impact.json').write_text(json.dumps(rows,indent=2)+'\n')
s=json.loads((R/'summary.json').read_text());lines=['# B300 kernel calibration follow-up','',f"Reused {s['reused_rows']} existing measurements; added {s['passed_measured_rows']} validated positive-latency measurements. Physical-key deduplication is recorded in deduplication.json. Native failures/refusals are retained in results/ and coverage.log.",'','| Input | Calibration | Predicted ms | Historical TPOT ms | Relative error | Measured share |','|---:|---|---:|---:|---:|---:|']
for x in rows:lines.append(f"| {x['input_tokens']} | {x['stage']} | {x['predicted_ms']:.4f} | {x['historical_tpot_ms']:.4f} | {x['error_pct']:+.2f}% | {x['partial_measured_share'] or 0:.2%} |")
lines+=['','The historical target-only serving measurements used three repeats, disabled speculation and excluded profiled requests. They are not AgentX or DFLASH measurements. See benchmark/dcp-validation in the results archive for full runtime provenance. This comparison reuses the recorded baseline; it does not claim a fresh end-to-end remeasurement.','', 'All three estimates above use the corrected branch (DCP formula reader, K3.1 fusion rules and explicit KV block size 64). Differences between existing and expanded isolate the overlay under those corrected model rules; differences from older report totals also include the rule/lookup corrections.','', 'The measured-envelope selector can choose a faster provider than SGLang dispatches. Cold-cache L2-drain timings differ from a warm CUDA graph; serial summation does not model full PDL/multi-stream overlap. Unsupported operator/collective gaps remain theoretical. No global fitted latency multiplier is applied.','', 'The previously suggested 5–8% error was a hypothesis. The table above, and only validated coverage in the artifacts, state what this run achieved. Partial coverage must not be described as a fully measured model.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
