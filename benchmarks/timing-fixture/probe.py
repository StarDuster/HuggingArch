import json,sys,dataclasses,datetime
from pathlib import Path
import torch
from kernel_bench.timing import time_callable
out=Path(__file__).resolve().parent
repeat=int(sys.argv[1])
p=torch.cuda.get_device_properties(0)
l2=getattr(p,'L2_cache_size',0) or getattr(p,'l2_cache_size',0)
assert isinstance(l2,int) and l2>0,l2
sizes=[16*2**20,3*l2//4]
# Predeclared alternation, never select a size from measured timing.
if repeat%2==0:sizes.reverse()
rows=[]
for size in sizes:
 gen=torch.Generator(device='cuda').manual_seed(2026111300+repeat)
 x=torch.randn(size//4,device='cuda',generator=gen)
 sink=torch.empty((),device='cuda')
 hot=time_callable(lambda:torch.amax(x,dim=0,out=sink),flush_l2=False)
 cold=time_callable(lambda:torch.amax(x,dim=0,out=sink))
 rows.append({'bytes':x.numel()*x.element_size(),'fraction_l2':x.numel()*x.element_size()/l2,
 'hot':dataclasses.asdict(hot),'cold':dataclasses.asdict(cold),'ratio':cold.p50_us/hot.p50_us})
 del x,sink
 torch.cuda.empty_cache()
result={'repeat':repeat,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device':p.name,'l2_bytes':l2,'torch':torch.__version__,'cuda':torch.version.cuda,'physical_gpu':7,'seed':2026111300+repeat,'measurements':rows}
(out/f'probe-{repeat}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
