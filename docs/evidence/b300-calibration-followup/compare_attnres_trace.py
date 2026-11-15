import json,gzip,statistics,csv,collections
from pathlib import Path
R=Path(__file__).resolve().parent;D=R.parent/'dcp-validation'
spans=json.loads((D/'continuous-trace-spans.json').read_text());rank=next(x for x in spans if x['rank']==0)
with gzip.open(rank['trace'],'rt') as f:trace=json.load(f)
events=trace.get('traceEvents',trace) if isinstance(trace,dict) else trace
kernels=[e for e in events if 'attn_res_fused_tma' in e.get('name','') and e.get('dur') is not None]
# Forward order from the model's history update: the segment-start entry
# joins the bank between that layer's attention and MLP combines.
sars=[1]
for layer in range(1,93):sars.extend([(layer+11)//12,layer//12+1])
sars.append(8);assert len(sars)==186
by=collections.defaultdict(list);checks=[]
for step in rank['steps']:
 if step.get('mode')!='decode':continue
 found=sorted([e for e in kernels if step['gpu_first_ts_us']<=e['ts']<step['gpu_last_ts_us']],key=lambda e:e['ts'])
 checks.append({'context':step['context'],'kernel_count':len(found),'expected':186})
 if len(found)!=186:continue
 for sar,e in zip(sars,found):by[(step['context'],sar)].append(e['dur'])
summary=[]
for (context,sar),values in sorted(by.items()):summary.append({'context':context,'Sar':sar,'samples':len(values),'median_us':statistics.median(values),'mean_us':statistics.mean(values),'p95_us':sorted(values)[int(.95*(len(values)-1))]})
report={'scope':'Historical TP0 CUPTI per-kernel durations in target-only CUDA graph; inferred Sar from verified 186-call forward order. These are diagnostics, not new calibration rows. Clock source differs from isolated CUDA-event measurements.','checks':checks,'summary':summary}
(R/'attnres-trace-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
