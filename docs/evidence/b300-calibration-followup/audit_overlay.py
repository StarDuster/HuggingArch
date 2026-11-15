import json,collections,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1];REPO=W/'calibration-worktree';PY=W/'HuggingArch/.venv313/bin/python'
rows=json.loads((R/'calibration_rows.json').read_text());existing=json.loads((R/'existing-calibration-rows.json').read_text())
# Preserve the broad cross-library corpus as evidence, but do not recommend
# eager references as a representative SGLang runtime overlay.
retained=list(existing);excluded=[]
for row in rows:
 if row in existing:continue
 if row['framework']!='torch':retained.append(row)
 else:excluded.append(row)
(R/'library_kernel_calibration_rows.json').write_text(json.dumps(retained,indent=2)+'\n')
expanded=json.loads((R/'estimates/16384-expanded.json').read_text());costs=collections.Counter()
for row in expanded['per_op_timing_decode']:costs[row['op']]+=row.get('t_measured_s') or 0
summary={'retained_rows':len(retained),'excluded_new_eager_rows':len(excluded),'rule':'Reuse the independently audited original 9 rows plus non-Torch library kernels. Keep all other data as cross-library reference measurements, not production runtime calibration. No latency threshold is used to choose rows.','measured_leaf_cost_ms_by_op':{k:1000*v for k,v in costs.most_common()},'source_version_fix':'git provenance used instead of unreleased SGLang placeholder 0.0.0','remaining_gaps':['MXFP4 grouped GEMM has no applicable native provider for these planned cells','TP8 collectives have no native measurement provider in this plan','AttnRes reference omits learned gain/projection inputs carried by the model graph; production kernel also fuses output RMSNorm, requiring a distinct span contract','isolated cold-cache serial timing differs from CUDA graph / PDL overlap']}
(R/'overlay-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
comparison=[]
for context in [2048,16384]:
 out=R/'estimates'/f'{context}-library-kernels.json'
 with (R/f'audit-estimate-{context}.log').open('w') as log:
  subprocess.run([str(PY),str(REPO/'scripts/estimate_kimi_k31_sglang.py'),'--dcp','8','--input-tokens',str(context),'--calibration-rows',str(R/'library_kernel_calibration_rows.json'),'--output',str(out)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
 data=json.loads(out.with_suffix('.summary.json').read_text())['phases']['decode'];base=next(r for r in json.loads((R/'estimate-impact.json').read_text()) if r['input_tokens']==context)
 comparison.append({'input_tokens':context,'partial_ms':1000*data['t_partial_s'],'historical_tpot_ms':base['historical_tpot_ms'],'error_pct':100*(1000*data['t_partial_s']/base['historical_tpot_ms']-1),'partial_measured_share':data['partial_measured_share'],'fully_measured_s':data['t_measured_s']})
(R/'audited-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
lines=['# Calibration audit and usable scope','',f"Retained {len(retained)} original/library-backed rows; excluded {len(excluded)} new eager reference rows from this supplementary overlay. The full raw corpus remains unchanged. Selection is based on implementation provenance, never closeness to the serving result.",'','| Input | Library-kernel partial estimate ms | Historical TPOT ms | Error |','|---:|---:|---:|---:|']
for x in comparison:lines.append(f"| {x['input_tokens']} | {x['partial_ms']:.4f} | {x['historical_tpot_ms']:.4f} | {x['error_pct']:+.2f}% |")
lines+=['','The first broad overlay predicted 74.34 ms. Correcting the source-version eligibility and measuring actual SGLang kernels reduced it to 47.32 ms. The remaining broad-overlay leaf costs include linear 12.876 ms, eager AttnRes 9.390 ms and eager router 7.834 ms; these are real timings of those providers, not evidence that SGLang pays them.','', 'The supplementary library overlay is partial, not an exact dispatch reconstruction. Unsupported MXFP4 expert kernels and TP8 collectives still use theory, AttnRes needs a weighted combine-plus-norm fusion contract, and cold isolated timings do not include serving overlap. A low aggregate error can result from cancellation and does not validate missing components. No 5–8% accuracy claim is supported by the available evidence.','', 'Use library_kernel_calibration_rows.json when inspecting the library-only partial overlay; calibration_rows.json retains the complete cross-library corpus for audit and should not be represented as production-specific SGLang calibration.']
(R/'CALIBRATION_AUDIT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
