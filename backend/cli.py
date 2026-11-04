"""HuggingArch CLI — analyze a model from the command line.

Thin wrapper around `backend.analyzer.analyze_model`. No torch needed
when a matching spec exists; the analyzer fetches config.json + safetensors
metadata + (when available) a snapshot, then runs the spec engine.

Usage:
    python -m backend.cli <model_id> [options]
    huggingarch <model_id> [options]   # via pyproject console script

Examples:
    python -m backend.cli facebook/opt-125m
    huggingarch deepseek-ai/DeepSeek-V2-Lite --seq-len 1024
    huggingarch Qwen/Qwen3-8B --output result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys


# Load environment variables from .env (project root, two levels up).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key, value)

from backend.analyzer import RESOLVED_KEY, analyze_model


def _print_summary(args, result: dict) -> None:
    summary = result.get("summary", {})

    print(f"\n📋 基础信息")
    print(f"  模型: {summary.get('model_id', args.model_id)}")
    print(f"  模型类型: {summary.get('model_type', 'N/A')}")
    print(f"  隐藏维度: {summary.get('hidden_size', 'N/A')}")
    print(f"  层数: {summary.get('num_layers', 'N/A')}")
    print(f"  词表大小: {summary.get('vocab_size', 'N/A')}")

    total_params = summary.get("total_params", 0)
    active_params = summary.get("total_active_params", total_params)
    print(f"\n📊 参数统计")
    print(f"  总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
    if active_params < total_params:
        print(f"  活跃参数量 (每Token): {active_params:,} ({active_params/1e9:.2f}B)")
    print(f"  模型大小: {summary.get('weight_bytes', 0) / 1024**3:.2f} GiB ({args.dtype})")

    total_flops = summary.get("total_flops_gflops", 0)
    print(f"\n⚡ FLOPs (Prefill, 含 Causal Mask)")
    print(f"  总FLOPs: {total_flops:.2f} GFLOPs")
    if total_flops > 0:
        attn_g = summary.get("attention_flops_gflops", 0)
        ffn_g = summary.get("ffn_flops_gflops", 0)
        print(f"  Attention: {attn_g:.2f} GFLOPs ({attn_g/total_flops*100:.1f}%)")
        print(f"  FFN: {ffn_g:.2f} GFLOPs ({ffn_g/total_flops*100:.1f}%)")

    attn_info = summary.get("attention_info", {})
    print(f"\n🎯 Attention配置")
    print(f"  类型: {(attn_info.get('attention_type') or 'N/A').upper()}")
    print(f"  Query Heads: {attn_info.get('num_heads', 'N/A')}")
    print(f"  KV Heads: {attn_info.get('num_kv_heads', 'N/A')}")
    print(f"  Head Dim: {attn_info.get('head_dim', 'N/A')}")
    if (attn_info.get("attention_type") or "").lower() == "mla":
        print(f"  Q Lora Rank: {attn_info.get('q_lora_rank', 'N/A')}")
        print(f"  KV Lora Rank: {attn_info.get('kv_lora_rank', 'N/A')}")

    ffn_info = summary.get("ffn_info", {})
    print(f"\n🔄 FFN配置")
    if ffn_info.get("ffn_type") == "moe":
        print(f"  类型: MoE")
        print(f"  专家数: {ffn_info.get('num_experts', 'N/A')}")
        print(f"  Top-k: {ffn_info.get('top_k', 'N/A')}")
        print(f"  MoE层数: {summary.get('num_moe_layers', 0)}")
        print(f"  Dense层数: {summary.get('num_dense_ffn_layers', 0)}")
    else:
        print(f"  类型: Dense")
        print(f"  Intermediate: "
              f"{ffn_info.get('intermediate_size', summary.get('intermediate_size', 'N/A'))}")

    layer_groups = result.get("layer_groups", [])
    limit = len(layer_groups) if args.verbose else 10
    title = "📑 层分组详情 (全部)" if args.verbose else f"📑 层分组详情 (前{limit}个)"
    print(f"\n{title}")
    for i, group in enumerate(layer_groups[:limit]):
        name = group.get("name", "N/A")
        count = group.get("count", 1)
        gp = group.get("total_params", 0)
        gflops = group.get("total_flops", 0)
        gactive = group.get("total_active_params", gp)
        active_str = f" (active: {gactive/1e6:.1f}M)" if gactive < gp else ""
        prefix = f"×{count}, " if count > 1 else ""
        print(f"  {i+1}. {name}: {prefix}{gp/1e6:.1f}M params{active_str}, "
              f"{gflops/1e9:.2f}G FLOPs")
    if len(layer_groups) > limit:
        print(f"  ... 还有 {len(layer_groups) - limit} 个层组 "
              f"(💡 加 --verbose 查看全部)")


def _bench_validate(argv: list[str]) -> int:
    sub = argparse.ArgumentParser(
        prog="huggingarch bench validate",
        description="Validate a benchmark CSV against the frozen schema "
                    "documented at backend/inference/BENCHMARKS_CSV.md.",
    )
    sub.add_argument("csv_path", help="path to the CSV file")
    sub.add_argument("--no-round-trip", action="store_true",
                     help="skip the round-trip byte-identity check")
    ns = sub.parse_args(argv)
    from backend.inference.benchmarks import (
        validate_file, assert_round_trip, BenchmarkValidationError,
    )
    try:
        _header, rows = validate_file(ns.csv_path)
        if not ns.no_round_trip:
            assert_round_trip(ns.csv_path)
    except BenchmarkValidationError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    print(f"✅ {ns.csv_path}: {len(rows)} row(s) valid.")
    return 0


def _bench_suggest_shapes(argv: list[str]) -> int:
    sub = argparse.ArgumentParser(
        prog="huggingarch bench suggest-shapes",
        description="Emit a benchmark CSV template populated from "
                    "AnalysisResult.per_op_timing. Identity columns are "
                    "filled where computable; measurement and provenance "
                    "columns are left blank for the contributor.",
    )
    sub.add_argument("model_id", help="HuggingFace model ID")
    sub.add_argument("--batch-size", type=int, default=1)
    sub.add_argument("--seq-len", type=int, default=2048)
    sub.add_argument("--framework",
                     help="fusion-binding framework (default: vllm)")
    sub.add_argument("--output", "-o",
                     help="write CSV to this path (default: stdout)")
    ns = sub.parse_args(argv)
    from backend.cli_bench import emit_suggest_shapes
    csv_text = emit_suggest_shapes(
        model_id=ns.model_id,
        batch_size=ns.batch_size,
        seq_len=ns.seq_len,
        **({"framework": ns.framework} if ns.framework else {}),
    )
    if ns.output:
        with open(ns.output, "w", encoding="utf-8") as f:
            f.write(csv_text)
        print(f"💾 wrote template to {ns.output}", file=sys.stderr)
    else:
        sys.stdout.write(csv_text)
    return 0


def _dispatch_bench(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: huggingarch bench {validate,suggest-shapes} ...",
            file=sys.stderr,
        )
        return 2
    sub = argv[0]
    rest = argv[1:]
    if sub == "validate":
        return _bench_validate(rest)
    if sub == "suggest-shapes":
        return _bench_suggest_shapes(rest)
    print(
        f"unknown bench subcommand {sub!r}; expected one of "
        f"{{validate, suggest-shapes}}.",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    # Route `huggingarch bench ...` to the bench subparser before the
    # legacy `<model_id>` flow consumes the first positional.
    if len(sys.argv) >= 2 and sys.argv[1] == "bench":
        return _dispatch_bench(sys.argv[2:])

    parser = argparse.ArgumentParser(
        description="HuggingArch — 分析 HuggingFace 模型架构 (无需下载权重)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_id", help="HuggingFace 模型 ID (如 facebook/opt-125m)")
    parser.add_argument("--batch-size", type=int, default=1, help="批大小 (默认 1)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="序列长度 (Prefill, 默认 2048)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="数据类型 (默认 float16)")
    parser.add_argument("--output", "-o", help="导出 JSON 文件路径")
    parser.add_argument("--no-trust-remote-code", action="store_true",
                        help="不信任远程代码 (某些模型可能需要)")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"🔍 HuggingArch")
    print(f"{'='*70}")
    print(f"模型: {args.model_id}")
    print(f"配置: batch={args.batch_size}, seq_len={args.seq_len}, dtype={args.dtype}")
    print(f"{'='*70}")

    try:
        result = analyze_model(
            model_id=args.model_id,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            dtype=args.dtype,
            trust_remote_code=not args.no_trust_remote_code,
        )
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        return 130
    except Exception as e:
        import traceback
        print(f"\n❌ 错误: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

    if not result.get("success"):
        print(f"\n❌ 分析失败: {result.get('error', '未知错误')}")
        if "message" in result:
            print(f"   详情: {result['message']}")
        return 1

    _print_summary(args, result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({key: value for key, value in result.items()
                       if key != RESOLVED_KEY},
                      f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 结果已导出到: {args.output}")

    print(f"\n{'='*70}\n✅ 分析完成\n{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
