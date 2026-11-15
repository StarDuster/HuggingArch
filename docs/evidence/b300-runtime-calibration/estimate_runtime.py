import json,sys
import statistics
import subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1];REPO=W/'calibration-worktree'
PY=W/'HuggingArch/.venv313/bin/python'
historical=json.loads((W/'benchmark/dcp-validation/results.json').read_text())['runs']
whole='--whole' in sys.argv
stem='whole-moe' if whole else 'runtime-span'
signature='kimi_k31_trtllm_moe_span' if whole else 'kimi_k31_trtllm_latent_span'
results=[]
for context in [16384,2048]:
    out=R/f'{"whole-" if whole else ""}prediction-{context}.json'
    print('Estimating',context,flush=True)
    with (R/f'prediction-{context}.log').open('w') as log:
        subprocess.run([str(PY),str(REPO/'scripts/estimate_kimi_k31_sglang.py'),
            '--dcp','8','--input-tokens',str(context),'--calibration-rows',str(R/(stem+'-overlay.json')),
            '--span-scope','whole' if whole else 'latent',
            '--kv-block-size','64','--output',str(out)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    phase=json.loads(out.with_suffix('.summary.json').read_text())['phases']['decode']
    detail=json.loads(out.read_text())
    span=[r for r in detail['per_op_timing_decode'] if r['op']==signature]
    assert sum(r['count_per_step'] for r in span)==92
    assert all(r['t_measured_s'] is not None for r in span), 'span did not match exact calibrated keys'
    baseline=statistics.mean(r['mean_tpot_ms'] for r in historical if r['input']==context)
    result=dict(input_tokens=context,partial_ms=phase['t_partial_s']*1000,baseline_ms=baseline,
        error_pct=100*(phase['t_partial_s']*1000/baseline-1),
        span_calls=92,span_ms=sum(r['t_measured_s'] for r in span)*1000,
        measured_share=phase['partial_measured_share'],fully_measured_s=phase['t_measured_s'])
    results.append(result)
    (R/('whole-comparison.json' if whole else 'runtime-comparison.json')).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(result),flush=True)
