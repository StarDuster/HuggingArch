"""Numerical and timing probe of the recipe's actual MXFP8/MXFP4 SiTU API.

The reference independently decodes E2M1 and computes routed FP32 GEMMs.
The API input is already MXFP8: route/input quantization is outside this span.
No result is admitted as calibration unless the unchanged native FP8 band passes.
"""
import json
import statistics
import time
from pathlib import Path
import torch
from flashinfer import autotune
from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
from flashinfer.tllm_enums import ActivationType, RoutingMethodType
from sglang.srt.layers.quantization.mxfp4 import (
    _prepare_flashinfer_mxfp8_activations,
    _get_flashinfer_mxfp4_device_permute_indices,
)
from kernel_bench.driver import _cross_check, _band_for
from kernel_bench.providers import BenchCase
from kernel_bench.quant import quantise_groups, FORMATS
from kernel_bench.gpulock import GpuLock

R = Path(__file__).resolve().parent
E, LOCAL_E, H, I, TOPK = 896, 448, 3584, 768, 16

def decode(packed):
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(-2)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device='cuda')
    return levels[(codes & 7).long()] * torch.where(codes < 8, 1., -1.) / 128

def reference(xq, xs, ids, scores, w1, w2, undo1, undo2, offset=0):
    x = xq.float().reshape(xq.shape[0], H // 32, 32)
    x = (x * torch.exp2(xs.view(torch.uint8).float().reshape(x.shape[0], H // 32, 1) - 127)).reshape(-1, H)
    out = torch.zeros_like(x)
    for expert in ids[(ids >= offset) & (ids < offset + LOCAL_E)].unique().tolist():
        tokens, slots = torch.where(ids == expert)
        first = decode(w1[expert - offset][undo1])
        second = decode(w2[expert - offset][undo2])
        # Source packs interleaved (up_i, gate_i), before the row shuffle.
        projected = x[tokens] @ first.T
        up, gate = projected[:, 0::2], projected[:, 1::2]
        hidden = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(up / 25))
        q, scales = quantise_groups(hidden, FORMATS['mxfp8'], 32, 'pow2', torch.float32)
        hidden = (q.float() * scales.unsqueeze(-1)).reshape(-1, I)
        expert_out = (hidden @ second.T).to(torch.bfloat16).float()
        out.index_add_(0, tokens, scores[tokens, slots, None] * expert_out)
    return out

def measure(fn, cold):
    # Both modes use CUDA Graph, so host dispatch is absent. The cold mode
    # evicts between calls; it does not charge an artificial post-kernel drain.
    evict = torch.zeros(256 << 20, dtype=torch.uint8, device='cuda') if cold else None
    for _ in range(5): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    repetitions = 1 if cold else 32
    with torch.cuda.graph(g):
        if cold: evict.zero_()
        start.record()
        for _ in range(repetitions): fn()
        end.record()
    samples = []
    for _ in range(30):
        g.replay(); end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repetitions)
    return dict(p50_us=statistics.median(samples), samples_us=samples,
                regime='graph-cold-before-only' if cold else 'graph-warm-repeat32')

def main():
    lock = GpuLock(); lock.acquire()
    results = []
    try:
        g = torch.Generator(device='cuda').manual_seed(125)
        print('Creating full local expert bank', flush=True)
        w1 = torch.randint(0, 256, (LOCAL_E, 2*I, H//2), dtype=torch.uint8, device='cuda', generator=g)
        w2 = torch.randint(0, 256, (LOCAL_E, H, I//2), dtype=torch.uint8, device='cuda', generator=g)
        sf1 = torch.full((LOCAL_E, 2*I, H//32), 120, dtype=torch.uint8, device='cuda').view(torch.float8_e4m3fn)
        sf2 = torch.full((LOCAL_E, H, I//32), 120, dtype=torch.uint8, device='cuda').view(torch.float8_e4m3fn)
        undo1 = torch.argsort(_get_flashinfer_mxfp4_device_permute_indices(w1[0], 128))
        undo2 = torch.argsort(_get_flashinfer_mxfp4_device_permute_indices(w2[0], 128))
        print('Permutation shapes', tuple(undo1.shape), tuple(undo2.shape), flush=True)
        for batch in [1, 8]:
            for seed in [17, 43]:
                g.manual_seed(seed + batch*1000)
                x = torch.randn(batch, H, dtype=torch.bfloat16, device='cuda', generator=g)
                ids = torch.rand(batch, E, device='cuda', generator=g).topk(TOPK, dim=-1).indices.to(torch.int32)
                scores = torch.rand(batch, TOPK, device='cuda', generator=g)
                scores /= scores.sum(-1, keepdim=True)
                _, _, xq, xs = _prepare_flashinfer_mxfp8_activations(x, H)
                output = torch.empty_like(x)
                kw = dict(topk_ids=(ids, scores), routing_bias=None, hidden_states=xq,
                    hidden_states_scale=xs, gemm1_weights=w1, gemm1_weights_scale=sf1,
                    gemm1_bias=None, gemm1_alpha=torch.full((LOCAL_E,), 4., device='cuda'),
                    gemm1_beta=torch.full((LOCAL_E,), 25., device='cuda'), gemm1_clamp_limit=None,
                    gemm2_weights=w2, gemm2_weights_scale=sf2, gemm2_bias=None,
                    output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None,
                    num_experts=E, top_k=TOPK, n_group=None, topk_group=None,
                    intermediate_size=I, local_expert_offset=0, local_num_experts=LOCAL_E,
                    routed_scaling_factor=None, routing_method_type=RoutingMethodType.TopK.value,
                    activation_type=ActivationType.Situ.value, tune_max_num_tokens=batch,
                    output=output, do_finalize=True, enable_pdl=True)
                fn = lambda: trtllm_fp4_block_scale_routed_moe(**kw)
                print('Checking', batch, seed, 'local assignments', int((ids<LOCAL_E).sum()), flush=True)
                with autotune(tune_mode=True): fn()
                ref = reference(xq, xs, ids, scores, w1, w2, undo1, undo2)
                case = BenchCase(run=fn, to_canonical=lambda unused: output)
                verdict, rel, cosine, typical, worst = _cross_check(case, ref, 'fp8_e4m3')
                band = _band_for('fp8_e4m3')
                result = dict(batch=batch, seed=seed, local_assignments=int((ids<LOCAL_E).sum()),
                              cross_check='passed' if verdict is None else 'failed', rel=rel,
                              cosine=cosine, typical=typical, worst=worst,
                              band=dict(typical=band.typical, worst=band.worst))
                print(json.dumps(result), flush=True)
                if verdict is None:
                    result['timing'] = [measure(fn, cold) for cold in [False, True]]
                    print('Timing', [(t['regime'], t['p50_us']) for t in result['timing']], flush=True)
                results.append(result)
                (R/'trtllm-probe-results.json').write_text(json.dumps(results, indent=2)+'\n')
                if verdict is not None:
                    raise RuntimeError('Native FP8 numerical band failed: no calibration admitted')
    finally:
        lock.release()

if __name__ == '__main__':
    main()
