"""
model_info — Extract model structure info for spec generation.

Usage:
    python -m backend.hf.model_info <model_id> [--output FILE]

Output: JSON with config, print(model), forward methods, and modeling source path.
No weights downloaded — uses init_empty_weights() for meta-tensor skeleton only.
"""
import argparse
import inspect
import json
import re
import sys
import os
import textwrap
from functools import lru_cache

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# The auto class extract_model_info instantiates the skeleton through.
SKELETON_AUTO_CLASS = "AutoModelForCausalLM"


@lru_cache(maxsize=1)
def causal_lm_heads() -> frozenset:
    """Architecture classes SKELETON_AUTO_CLASS builds for a config, from
    the installed Transformers' own auto mapping."""
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    )
    return frozenset(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values())


def extract_model_info(model_id: str, trust_remote_code: bool = True) -> dict:
    """Load model skeleton and extract structural info."""
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights

    # Load config
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    config_dict = config.to_dict()

    # Load skeleton (eager attention — we only inspect structure, not run inference)
    def _set_eager(cfg):
        if hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
        if hasattr(cfg, "_attn_implementation_internal"):
            cfg._attn_implementation_internal = "eager"
        for attr in vars(cfg):
            sub = getattr(cfg, attr, None)
            if hasattr(sub, "_attn_implementation"):
                _set_eager(sub)
    _set_eager(config)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code,
            attn_implementation="eager",
        )

    model_str = str(model)

    # Extract per-module info
    modules = {}
    for name, module in model.named_modules():
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        total = sum(p.numel() for p in params)
        shape = list(params[0].shape) if params else []
        modules[name] = {
            "class": module.__class__.__name__,
            "params": total,
            "shape": shape,
        }
        if hasattr(module, "in_features"):
            modules[name]["in_features"] = module.in_features
        if hasattr(module, "out_features"):
            modules[name]["out_features"] = module.out_features
        if isinstance(module, nn.Embedding):
            modules[name]["num_embeddings"] = module.num_embeddings
            modules[name]["embedding_dim"] = module.embedding_dim

    total_params = sum(m["params"] for m in modules.values())

    # Detect block repetition pattern
    block_pattern = _detect_block_pattern(modules)

    # Extract modeling source and forward methods
    source_info = _extract_source_info(model)

    return {
        "model_id": model_id,
        "config": config_dict,
        "model_str": model_str,
        "modules": modules,
        "total_params": total_params,
        "block_pattern": block_pattern,
        "source": source_info,
    }


def _extract_source_info(model) -> dict:
    """Extract modeling source path and key forward methods."""
    result = {
        "modeling_file": None,
        "forward_methods": {},
    }

    # Find the modeling file from the model class
    model_cls = type(model)
    try:
        result["modeling_file"] = inspect.getfile(model_cls)
    except (TypeError, OSError):
        pass

    # One excerpt per distinct module class inside the representative layers
    # (one per child-class signature, so every mixer of a hybrid is walked),
    # labelled by its own class name. Nothing is picked by what its name
    # says: a GatedDeltaNet or Mamba mixer is as much the layer's wiring as
    # an `*Attention`, and a model with two attention classes has two
    # forwards. PyTorch's own building blocks (Linear, Embedding, ModuleList)
    # are not the model's code and carry no wiring of its.
    classes_to_extract = {"Model": model_cls}
    for layer in _find_all_layers(model):
        for _name, module in layer.named_modules():
            cls = type(module)
            if cls.__module__.split(".")[0] == "torch":
                continue
            label = cls.__name__
            if classes_to_extract.get(label, cls) is not cls:
                label = f"{cls.__module__}.{cls.__qualname__}"
            classes_to_extract.setdefault(label, cls)

    # Extract forward methods
    for label, cls in classes_to_extract.items():
        try:
            src = inspect.getsource(cls.forward)
            src = textwrap.dedent(src)
            result["forward_methods"][label] = src
        except (TypeError, OSError):  # per-item: one bad item must not abort the loop
            continue

    return result


def _find_all_layers(model) -> list:
    """Find all transformer decoder layers. Returns a deduplicated sample (first + any structurally different)."""
    import torch.nn as nn
    layers_list = None

    # Common patterns: model.model.layers, model.transformer.h, etc.
    for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                layers_list = obj
                break
        except AttributeError:
            continue

    # Fallback: find any ModuleList whose children have attention
    if layers_list is None:
        for _name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                child = module[0]
                if any("attn" in n for n, _ in child.named_children()):
                    layers_list = module
                    break

    if layers_list is None:
        return []

    # Deduplicate: return one representative per unique class-signature
    seen_sigs = set()
    result = []
    for layer in layers_list:
        sig = tuple(type(child).__name__ for _, child in layer.named_children())
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            result.append(layer)
    return result


def _detect_block_pattern(modules: dict) -> dict:
    """Detect repeated transformer block pattern from module names."""
    layer_modules = {}
    layer_re = re.compile(r"^(.+?)\.(\d+)\.(.+)$")

    for name in modules:
        m = layer_re.match(name)
        if m:
            idx, suffix = int(m.group(2)), m.group(3)
            layer_modules.setdefault(idx, set()).add(suffix)

    if not layer_modules:
        return {"repeat": 0, "modules": []}

    patterns = {}
    for idx, mods in layer_modules.items():
        key = frozenset(mods)
        patterns.setdefault(key, []).append(idx)

    groups = []
    for pattern, indices in sorted(patterns.items(), key=lambda x: min(x[1])):
        groups.append({
            "repeat": len(indices),
            "range": [min(indices), max(indices)],
            "modules": sorted(pattern),
        })

    if len(groups) == 1:
        return groups[0]

    return {"groups": groups, "total_layers": len(layer_modules)}


def main():
    parser = argparse.ArgumentParser(description="Extract model info for spec generation")
    parser.add_argument("model_id", help="HuggingFace model ID")
    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()

    info = extract_model_info(args.model_id, trust_remote_code=not args.no_trust_remote_code)

    output = json.dumps(info, indent=2, ensure_ascii=False, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Side-effect: populate the shared forward-source cache so callers
    # like validate_spec can pick up high-fidelity forward methods
    # without re-running this subprocess. Best-effort — any write
    # failure is silently ignored (cache is an optimization, not a
    # correctness requirement).
    try:
        from backend.hf.forward_source import model_info_cache_path
        cache = model_info_cache_path(args.model_id)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(output)
    except (ImportError, OSError):
        pass


if __name__ == "__main__":
    main()
