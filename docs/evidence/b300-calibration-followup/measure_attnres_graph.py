"""Explicit CUDA-graph warm measurement variant; retain cold rows separately."""
import json,statistics,datetime,dataclasses
from pathlib import Path
import torch
from kernel_bench import driver,providers
from kernel_bench.env import detect_env
from kernel_bench.providers import LatencyStats
from kernel_bench.emit import rows_to_csv
from kernel_bench.gpulock import GpuLock
from backend.inference.benchmarks import validate_text
R=Path(__file__).resolve().parent

def graph_time(call):
 for _ in range(10):call()
 torch.cuda.synchronize()
 g=torch.cuda.CUDAGraph();repeats=64
 with torch.cuda.graph(g):
  for _ in range(repeats):call()
 for _ in range(5):g.replay()
 torch.cuda.synchronize()
 samples=[];start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 for _ in range(30):
  start.record();g.replay();end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000/repeats)
 return LatencyStats(statistics.median(samples),sorted(samples)[28],statistics.mean(samples),30,'warm_l2')
providers.discover()
providers._PROVIDERS[:]=[dataclasses.replace(p,kernel_impl=p.kernel_impl+'__cuda_graph_warm') if p.signature=='kimi_attn_res_rmsnorm' else p for p in providers._PROVIDERS]
driver.time_callable=graph_time
env=detect_env();env.library='sglang'
jobs=[json.loads(l) for l in (R/'weighted-attn-res-jobs.jsonl').read_text().splitlines()]
lock=GpuLock();lock.acquire()
try:
 result=driver.run_jobs(jobs,env=env,date=datetime.date.today().isoformat(),contributor='StarDuster explicit CUDA Graph warm64 variant',run_id='b300-attnres-graph-warm',box='B300 GPU7, graph64, no L2 drain',cross_check=True,log=lambda s:print(s,flush=True))
 text=rows_to_csv(result.rows);validate_text(text)
 (R/'results/sglang-weighted-attn-res-graph-warm.csv').write_text(text)
 (R/'attnres-graph-result.json').write_text(json.dumps(dataclasses.asdict(result),indent=2)+'\n')
 print('rows',len(result.rows),'failures',result.failures,flush=True)
finally:lock.release()
