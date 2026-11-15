"""Actual K3 TP8/EP2 MoE block, with shared branch on its source side stream.

BF16 hidden -> replicated FP32 merged front -> K3 routing/quant/metadata ->
deferred TRTLLM experts; parallel shared SiTU/GEMM/low-SM pull AR; join ->
deferred finalize/push AR/norm -> column-sharded GEMV/all-gather/add.
Four independent dense-weight sets and sixteen routes prevent hot-weight
repetition. No checkpoint, HTTP server, or fitted overlap coefficient.
"""
import os,json,statistics
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import torch
import torch.distributed as dist
from kernel_bench.gpulock import GpuLock
from kernel_bench.driver import _cross_check
from kernel_bench.providers import BenchCase
from kernel_bench.quant import quantise_groups,FORMATS
from sglang.kernels.ops.communication import all_reduce_residual as ar,gemm_ag
from sglang.kernels.ops.activation import situ_and_mul
from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm_out
from sglang.kernels.ops.moe import kimi_k3_fused_routing as route
from sglang.srt.layers.quantization.mxfp4 import _get_flashinfer_mxfp4_device_permute_indices
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import _run_trtllm_prepared_routing_moe,_trtllm_moe_default_tile_n
from flashinfer.fused_moe.da_moe import DABody
from trtllm_moe_probe import reference

R=Path(__file__).resolve().parent
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==8
torch.cuda.set_device(local);lock=GpuLock(device=local);lock.acquire()
H,D,I,E,LE,K,SI=3584,7168,768,896,448,16,768
offset=(rank//4)*LE
def log(*a):print(f'rank{rank}:',*a,flush=True)

try:
    dist.init_process_group('gloo',timeout=timedelta(seconds=240))
    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
    coord=ps.init_world_group(ranks=list(range(8)),local_rank=local,backend='nccl');ps._WORLD=coord
    get_parallel().override_permanently(world_group=coord)
    comm=CustomAllReduceV2(coord.cpu_group,torch.device('cuda',local),max_pull_size=4*1024**2,max_push_size=4*1024**2)
    assert not comm.disabled and comm.has_multicast;ar.register_comm(comm.obj)
    # The pull AR's exact source allocation contract, including multicast VA.
    from torch._C._distributed_c10d import _SymmetricMemory
    shared_buffer=_SymmetricMemory.empty_strided_p2p((8,D),[D,1],torch.bfloat16,torch.device('cuda',local),coord.cpu_group.group_name)
    shared_handle=_SymmetricMemory.rendezvous(shared_buffer)
    assert shared_handle is not None and shared_handle.multicast_ptr
    g=torch.Generator(device='cuda').manual_seed(125+rank*1000)
    w1=torch.randint(0,256,(LE,2*I,H//2),device='cuda',dtype=torch.uint8,generator=g)
    w2=torch.randint(0,256,(LE,H,I//2),device='cuda',dtype=torch.uint8,generator=g)
    weights=SimpleNamespace(w13_weight=w1,w2_weight=w2,
        w13_weight_scale=torch.full((LE,2*I,H//32),120,device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn),
        w2_weight_scale=torch.full((LE,H,I//32),120,device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn),
        gemm1_alpha=torch.full((LE,),4.,device='cuda'),gemm1_clamp_limit=torch.full((LE,),25.,device='cuda'),
        global_num_experts=E,local_num_experts=LE,local_expert_offset=offset,intermediate_size_per_partition=I)
    undo1=torch.argsort(_get_flashinfer_mxfp4_device_permute_indices(w1[0],128))
    undo2=torch.argsort(_get_flashinfer_mxfp4_device_permute_indices(w2[0],128))
    dense=[]
    for slot in range(4):
        # Replicated router/latent down/up/norm; rank-specific shared shards.
        g.manual_seed(2231+slot*100)
        router_latent=(torch.randn(E+H,D,device='cuda',dtype=torch.bfloat16,generator=g)/D**.5).contiguous()
        up=(torch.randn(D,H,device='cuda',dtype=torch.bfloat16,generator=g)/H**.5).contiguous()
        norm=(1+torch.randn(H,device='cuda',dtype=torch.bfloat16,generator=g)*.05).contiguous()
        bias=torch.randn(E,device='cuda',dtype=torch.float32,generator=g)*.05
        g.manual_seed(3341+slot*100+rank*10000)
        shared_gu=(torch.randn(2*SI,D,device='cuda',dtype=torch.bfloat16,generator=g)/D**.5).contiguous()
        shared_down=(torch.randn(D,SI,device='cuda',dtype=torch.bfloat16,generator=g)/ (8*SI)**.5).contiguous()
        front=torch.cat([shared_gu,router_latent],0).contiguous()
        dense.append(dict(front=front,up=up,norm=norm,bias=bias,shared_down=shared_down))
    log('weights and multicast storage ready')
    side=torch.cuda.Stream();main=torch.cuda.current_stream()
    results=[]
    for batch in [1,8]:
        body=DABody(tactic=-1,tile_n=_trtllm_moe_default_tile_n(batch,K,LE))
        latent=torch.empty(batch,H,dtype=torch.bfloat16,device='cuda')
        shared=shared_buffer[:batch]
        output=torch.empty(batch,D,dtype=torch.bfloat16,device='cuda')
        front_workspace=torch.empty(batch,2*SI+E+H,device='cuda',dtype=torch.float32)
        activation_workspace=torch.empty(batch,SI,device='cuda',dtype=torch.bfloat16)
        def run(case,parallel=True):
            main=torch.cuda.current_stream()
            worker=side if parallel else main
            x,slot=case;p=dense[slot]
            front=front_workspace
            cutedsl_bf16_gemm_out(x,p['front'],front)
            gate_up=front[:,:2*SI];scores=front[:,2*SI:2*SI+E];routed_input=front[:,2*SI+E:]
            worker.wait_stream(main)
            routed=route.fused_routing(scores,p['bias'],routed_input,K,True,1.,True,
                num_token_non_padded=None,local_expert_offset=offset,num_local_experts=LE,
                tile_tokens=body.tile_n,weights_dtype=route.expert_weights_dtype(batch))
            partial=_run_trtllm_prepared_routing_moe(weights,routed.ids,routed.metadata,body,routed.x_q,
                routed.x_s.view(torch.float8_e4m3fn),latent,False)
            with torch.cuda.stream(worker):
                activated=situ_and_mul(gate_up,activation_workspace,4.,25.)
                cutedsl_bf16_gemm_out(activated,p['shared_down'],shared)
                ar.all_reduce_pull_res(8,shared,None,input_mc_ptr=shared_handle.multicast_ptr,num_blocks=4,unroll=8)
            main.wait_stream(worker)
            gemm2,probs,indices=partial[:3]
            ar.finalize_all_reduce_push_norm(8,latent,gemm2,indices,probs,p['norm'],1e-6)
            return gemm_ag.gemm_ag_up_proj(8,latent,p['up'],shared,None,output)
        def oracle(case):
            x,slot=case;p=dense[slot]
            front=x.float()@p['front'].float().T
            gate,up=front[:,:SI],front[:,SI:2*SI]
            shared_hidden=((4*torch.tanh(gate/4)*torch.sigmoid(gate))*(25*torch.tanh(up/25))).to(torch.bfloat16)
            shared_ref=(shared_hidden.float()@p['shared_down'].float().T).to(torch.bfloat16).float().cpu()
            dist.all_reduce(shared_ref);shared_ref=shared_ref.to(torch.bfloat16).float().cuda()
            scores=front[:,2*SI:2*SI+E];x_latent=front[:,2*SI+E:]
            raw=torch.sigmoid(scores);ids=(raw+p['bias']).topk(K,dim=-1).indices.to(torch.int32)
            probs=raw.gather(-1,ids.long());probs/=probs.sum(-1,keepdim=True)
            if route.expert_weights_dtype(batch)==torch.bfloat16:probs=probs.to(torch.bfloat16).float()
            q,scales=quantise_groups(x_latent,FORMATS['mxfp8'],32,'pow2',torch.float32)
            scale_bytes=(torch.log2(scales).to(torch.int32)+127).to(torch.uint8)
            partial=reference(q.reshape(batch,H),scale_bytes,ids,probs,w1,w2,undo1,undo2,offset)
            global_latent=partial.cpu();dist.all_reduce(global_latent)
            global_latent=global_latent.to(torch.bfloat16).float().cuda()
            normalized=(global_latent*torch.rsqrt(global_latent.square().mean(-1,keepdim=True)+1e-6)*p['norm'].float()).to(torch.bfloat16)
            projected=(normalized.float()@p['up'].float().T).to(torch.bfloat16).float()
            return (projected+shared_ref).to(torch.bfloat16)
        checks=[]
        for index,seed in enumerate([17,43]):
            g.manual_seed(seed+batch*1000)
            case=(torch.randn(batch,D,device='cuda',dtype=torch.bfloat16,generator=g),index)
            expected=oracle(case);dist.barrier()
            verdict,rel,cos,typ,worst=_cross_check(BenchCase(run=lambda:run(case),to_canonical=lambda y:y),expected,'fp8_e4m3')
            checks.append(dict(seed=seed,passed=verdict is None,rel=rel,cosine=cos,typical=typ,worst=worst))
            log('numeric',batch,checks[-1]);assert verdict is None,checks[-1]
        cases=[]
        for i in range(16):
            g.manual_seed(731+batch*1000+i*199)
            cases.append((torch.randn(batch,D,device='cuda',dtype=torch.bfloat16,generator=g),i%4))
        for case in cases:run(case)
        torch.cuda.synchronize();dist.barrier()
        variants={}
        for parallel in [True,False]:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for case in cases:run(case,parallel)
            for _ in range(3):graph.replay()
            torch.cuda.synchronize();dist.barrier()
            samples=[];start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            for _ in range(30):
                dist.barrier();start.record();graph.replay();end.record();end.synchronize()
                samples.append(start.elapsed_time(end)*1000/len(cases))
            variants['two_streams' if parallel else 'serial_control']=dict(p50_us=statistics.median(samples),samples_us=samples)
        samples=variants['two_streams']['samples_us']
        result=dict(rank=rank,batch=batch,checks=checks,p50_us=statistics.median(samples),samples_us=samples,
            timing_method='cuda_graph_whole_moe_two_streams_routes16_weights4',weight_sets=4,rotating_routes=16,
            scope='complete K3 TP8/EP2 latent MoE block including replicated FP32 dense front, shared side stream, deferred experts, collectives, norm, GEMV/allgather/add; prefix_sum=None',
            mesh=dict(tp=8,ep=2,expert_tp=4))
        result['variants']=variants
        results.append(result);(R/f'whole-moe-rank{rank}.json').write_text(json.dumps(results,indent=2)+'\n')
        log('whole MoE measured',batch,result['p50_us'])
    dist.barrier();comm.close();dist.destroy_process_group()
finally:lock.release()
