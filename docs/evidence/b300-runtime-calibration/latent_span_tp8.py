"""Exact recipe subpath: prepared K3 routing -> deferred TRTLLM -> AR+norm.

All eight ranks participate. Numerical reference is decoded FP32 expert
math + independent CPU Gloo reduction, then FP32 RMSNorm. Timing rotates
16 independent routes so the same eight expert matrices are not repeated
in L2. No global serving-time factor is fitted.
"""
import os
import json
import statistics
import datetime
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import torch
import torch.distributed as dist
from kernel_bench.gpulock import GpuLock
from kernel_bench.driver import _cross_check
from kernel_bench.providers import BenchCase
from sglang.kernels.ops.moe import kimi_k3_fused_routing as route
from sglang.kernels.ops.communication import all_reduce_residual as ar
from sglang.srt.layers.quantization.mxfp4 import _get_flashinfer_mxfp4_device_permute_indices
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    _run_trtllm_prepared_routing_moe, _trtllm_moe_default_tile_n,
)
from flashinfer.fused_moe.da_moe import DABody
from trtllm_moe_probe import reference

R = Path(__file__).resolve().parent
rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
world = int(os.environ['WORLD_SIZE']); assert world == 8
torch.cuda.set_device(local)
lock = GpuLock(device=local); lock.acquire()
H, I, E, LE, K = 3584, 768, 896, 448, 16
offset = (rank // 4) * LE

def log(*values): print(f'rank{rank}:', *values, flush=True)

def sample(batch, seed):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    x = torch.randn(batch, H, device='cuda', dtype=torch.bfloat16, generator=gen)
    scores = torch.randn(batch, E, device='cuda', dtype=torch.float32, generator=gen)
    bias = torch.randn(E, device='cuda', dtype=torch.float32, generator=gen) * .05
    return x, scores, bias

def routing(x, scores, bias, body):
    return route.fused_routing(scores, bias, x, K, True, 1.0, True,
        num_token_non_padded=None, local_expert_offset=offset,
        num_local_experts=LE, tile_tokens=body.tile_n,
        weights_dtype=route.expert_weights_dtype(x.shape[0]))

try:
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
    coord = ps.init_world_group(ranks=list(range(world)), local_rank=local, backend='nccl')
    ps._WORLD = coord
    get_parallel().override_permanently(world_group=coord)
    comm = CustomAllReduceV2(coord.cpu_group, torch.device('cuda', local), max_pull_size=4*1024**2, max_push_size=4*1024**2)
    assert not comm.disabled and comm.has_multicast
    ar.register_comm(comm.obj)
    log('multicast ready; creating independent expert partition')
    gen = torch.Generator(device='cuda').manual_seed(125 + rank*1000)
    w1 = torch.randint(0,256,(LE,2*I,H//2),dtype=torch.uint8,device='cuda',generator=gen)
    w2 = torch.randint(0,256,(LE,H,I//2),dtype=torch.uint8,device='cuda',generator=gen)
    weights = SimpleNamespace(
        w13_weight=w1, w2_weight=w2,
        w13_weight_scale=torch.full((LE,2*I,H//32),120,dtype=torch.uint8,device='cuda').view(torch.float8_e4m3fn),
        w2_weight_scale=torch.full((LE,H,I//32),120,dtype=torch.uint8,device='cuda').view(torch.float8_e4m3fn),
        gemm1_alpha=torch.full((LE,),4.,device='cuda'),
        gemm1_clamp_limit=torch.full((LE,),25.,device='cuda'),
        global_num_experts=E, local_num_experts=LE, local_expert_offset=offset,
        intermediate_size_per_partition=I)
    undo1 = torch.argsort(_get_flashinfer_mxfp4_device_permute_indices(w1[0],128))
    undo2 = torch.argsort(_get_flashinfer_mxfp4_device_permute_indices(w2[0],128))
    gen.manual_seed(731)
    norm = (1 + torch.randn(H,dtype=torch.bfloat16,device='cuda',generator=gen)*.05).contiguous()
    results=[]
    for batch in [1,8]:
        body = DABody(tactic=-1,tile_n=_trtllm_moe_default_tile_n(batch,K,LE))
        out = torch.empty(batch,H,dtype=torch.bfloat16,device='cuda')
        def run(args):
            x,scores,bias=args
            routed=routing(x,scores,bias,body)
            # Same zero-copy dtype view as _prepare_flashinfer_mxfp8_activations.
            partial=_run_trtllm_prepared_routing_moe(weights,routed.ids,routed.metadata,body,routed.x_q,routed.x_s.view(torch.float8_e4m3fn),out,False)
            gemm2,expert_weights,indices=partial[:3]
            ar.finalize_all_reduce_push_norm(8,out,gemm2,indices,expert_weights,norm,1e-6)
            return out
        checks=[]
        for seed in [17,43]:
            args=sample(batch,seed+batch*1000)
            x,scores,bias=args
            routed=routing(x,scores,bias,body)
            # Independent routing oracle; FP32 scores determine winners.
            raw=torch.sigmoid(scores.float())
            ids=(raw+bias).topk(K,dim=-1).indices.to(torch.int32)
            assert torch.equal(ids.sort(-1).values,routed.ids.sort(-1).values)
            probs=raw.gather(-1,ids.long());probs/=probs.sum(-1,keepdim=True)
            if route.expert_weights_dtype(batch)==torch.bfloat16:
                probs=probs.to(torch.bfloat16).float()
            partial_ref=reference(routed.x_q,routed.x_s,ids,probs,w1,w2,undo1,undo2,offset)
            expected=partial_ref.cpu()
            dist.all_reduce(expected)
            # Production AR stores its sum in BF16 before the norm.
            expected=expected.to(torch.bfloat16).float()
            expected=expected*torch.rsqrt(expected.square().mean(-1,keepdim=True)+1e-6)*norm.float().cpu()
            expected=expected.cuda()
            dist.barrier()
            verdict,rel,cos,typ,worst=_cross_check(BenchCase(run=lambda:run(args),to_canonical=lambda y:y),expected,'fp8_e4m3')
            checks.append(dict(seed=seed,passed=verdict is None,rel=rel,cosine=cos,typical=typ,worst=worst))
            log('numeric',batch,checks[-1])
            assert verdict is None, checks[-1]
        # Every call rereads different routed expert weights; no flush/drain
        # kernels are inserted between stages of the actual execution span.
        cases=[sample(batch,311+batch*1000+i*199) for i in range(16)]
        for args in cases:run(args)
        torch.cuda.synchronize();dist.barrier()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for args in cases:run(args)
        for _ in range(3):graph.replay()
        torch.cuda.synchronize();dist.barrier()
        samples=[];start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        for _ in range(30):
            dist.barrier();start.record();graph.replay();end.record();end.synchronize()
            samples.append(start.elapsed_time(end)*1000/len(cases))
        result=dict(rank=rank,batch=batch,checks=checks,p50_us=statistics.median(samples),samples_us=samples,
                    timing_method='cuda_graph_rotating_routes16',tile_n=body.tile_n,tactic=body.tactic,
                    scope='K3 prepared routing + input MXFP8 quant + TRTLLM deferred expert GEMMs + finalize/TP8 AR/RMSNorm; excludes dense front/shared branch/latent up',
                    mesh=dict(tp=8,ep=2,expert_tp=4),rotating_routes=16,reference='independent decoded FP32 expert GEMMs + CPU Gloo reduction + RMSNorm')
        results.append(result)
        (R/f'latent-span-rank{rank}.json').write_text(json.dumps(results,indent=2)+'\n')
        log('measured',batch,result['p50_us'])
    dist.barrier();comm.close();dist.destroy_process_group()
finally:
    lock.release()
