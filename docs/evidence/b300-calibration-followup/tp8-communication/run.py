"""TP8 SGLang K3 multicast all-reduce: exact arithmetic + steady CUDA graph timing.
Initialization follows the pinned source's test_collectives.py::_init_comm.
No model weights, serving process or synthetic latency multiplier.
"""
import os,json,time,statistics,datetime,subprocess
from pathlib import Path
import torch
import torch.distributed as dist
from datetime import timedelta
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
assert world==8
torch.cuda.set_device(local)
from kernel_bench.gpulock import GpuLock
lock=GpuLock(device=local);lock.acquire()
import sglang.srt.distributed.parallel_state as ps
from sglang.srt.runtime_context import get_parallel
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
from sglang.kernels.ops.communication import all_reduce_residual as ar
R=Path(__file__).resolve().parent

def log(s):
 print(f'rank{rank}: {s}',flush=True)

try:
 dist.init_process_group('gloo',timeout=timedelta(seconds=180))
 coord=ps.init_world_group(ranks=list(range(world)),local_rank=local,backend='nccl')
 ps._WORLD=coord
 get_parallel().override_permanently(world_group=coord)
 comm=CustomAllReduceV2(coord.cpu_group,torch.device('cuda',local),max_pull_size=4*1024**2,max_push_size=4*1024**2)
 assert not comm.disabled and comm.has_multicast,'multicast unavailable; do not silently substitute NCCL'
 ar.register_comm(comm.obj)
 log('multicast communicator ready')
 results=[]
 for n in [3584,7168,28672,57344]:
  # Two independent integer-input checks. Integers keep the BF16 sums exact.
  for seed in [17,43]:
   gen=torch.Generator(device='cuda').manual_seed(seed+rank*1000+n)
   x=torch.randint(0,16,(n,),device='cuda',dtype=torch.int32,generator=gen).to(torch.bfloat16)
   expected=x.float()
   # Explicit CPU/Gloo FP32 oracle, independent from the CUDA collective.
   oracle=expected.cpu();dist.all_reduce(oracle)
   ar.all_reduce_push_res(world,x,None);torch.cuda.synchronize()
   torch.testing.assert_close(x.float().cpu(),oracle,rtol=0,atol=0)
  x=torch.zeros(n,device='cuda',dtype=torch.bfloat16)
  for _ in range(20):ar.all_reduce_push_res(world,x,None)
  torch.cuda.synchronize();dist.barrier()
  # Zero input stays zero across repeated in-place reductions. There is no
  # data-dependent shortcut in this kernel; random-input correctness is above.
  graph=torch.cuda.CUDAGraph();repeats=128
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   with torch.cuda.graph(graph,stream=stream):
    for _ in range(repeats):ar.all_reduce_push_res(world,x,None)
  torch.cuda.current_stream().wait_stream(stream)
  torch.cuda.synchronize();dist.barrier()
  for _ in range(5):graph.replay()
  torch.cuda.synchronize();dist.barrier()
  samples=[];start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  for _ in range(30):
   dist.barrier();start.record();graph.replay();end.record();end.synchronize()
   samples.append(start.elapsed_time(end)*1000/repeats)
  assert torch.count_nonzero(x).item()==0
  result={'rank':rank,'n_comm_ranks':world,'n_comm_elems':n,'dtype':'bf16','residual':False,'seeds':[17,43],'cross_check':'passed','oracle':'Gloo FP32 exact integer sums','timing_method':'warm_l2','timing_scope':'steady CUDA graph, 128 reductions/replay; input/output resident, no L2 drain','graph_repeats':repeats,'p50_us':statistics.median(samples),'p95_us':sorted(samples)[28],'samples_us':samples}
  results.append(result);log(f'n={n} p50={result["p50_us"]:.3f}us exact-check passed')
 (R/f'rank{rank}.json').write_text(json.dumps(results,indent=2)+'\n')
 dist.barrier()
 if rank==0:
  all_results=[]
  for r in range(world):all_results.extend(json.loads((R/f'rank{r}.json').read_text()))
  summary=[]
  for n in [3584,7168,28672,57344]:
   vals=[r for r in all_results if r['n_comm_elems']==n]
   summary.append({'n_comm_elems':n,'n_comm_ranks':8,'dtype':'bf16','rank_p50_us':[v['p50_us'] for v in vals],'max_rank_p50_us':max(v['p50_us'] for v in vals),'cross_check':'passed','timing_method':'warm_l2','scope':'SGLang multicast all_reduce_push_res with residual=None; not a fused AR+norm measurement'})
  (R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary),flush=True)
 dist.barrier();comm.close();dist.destroy_process_group()
finally:
 lock.release()
