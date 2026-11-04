"""Custom model registration for non-HuggingFace sources.

Lets users analyze models whose weights / code don't live on the HF hub —
e.g. internal checkpoints, research prototypes, forks. Inputs are a local
config.json + one or more modeling*.py source files + optional safetensors
index. Output is a cached spec in `~/.huggingarch/custom_models/<name>/`
that the normal analyze pipeline picks up transparently via the
`custom:<name>` model_id protocol.

Cache layout per model:
    ~/.huggingarch/custom_models/<name>/
        config.json            — verbatim copy
        forward.py             — concatenated modeling source
        spec.yaml              — auto-generated arch spec
        checkpoint_header.json — the safetensors header, verbatim (only when
                                 registered with `--hf-model-path`)
        meta.json              — created/updated timestamps, source paths

The cache dir is outside the project tree and never touched by git.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.arch.quant_schemes import resolve_scheme
from backend.custom_id import (
    NAME_PREFIX,
    cache_dir,
    model_path,
    strip_prefix,
)


def list_all() -> List[str]:
    root = cache_dir()
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "config.json").exists())


def _next_auto_name() -> str:
    existing = set(list_all())
    n = 1
    while True:
        name = f"custom_model_{n}"
        if name not in existing:
            return name
        n += 1


def _safe_name(name: str) -> str:
    if not re.match(r"^[A-Za-z0-9_\-]+$", name):
        raise ValueError(f"custom model name must match [A-Za-z0-9_-]+, got {name!r}")
    return name


def load(name: str) -> Dict[str, Any]:
    """Load a cached custom model. Returns config, forward source, the
    checkpoint header (maybe None), spec path. Raises FileNotFoundError if not
    registered."""
    d = model_path(name)
    if not d.is_dir():
        raise FileNotFoundError(f"custom model not registered: {name} (expected at {d})")

    config = json.loads((d / "config.json").read_text())
    forward_src = (d / "forward.py").read_text() if (d / "forward.py").exists() else ""
    spec_yaml = (d / "spec.yaml").read_text() if (d / "spec.yaml").exists() else ""
    header = None
    if (d / "checkpoint_header.json").exists():
        from backend.hf.header import SafetensorsHeader
        header = SafetensorsHeader.from_json(json.loads(
            (d / "checkpoint_header.json").read_text())["header"])
    meta = {}
    if (d / "meta.json").exists():
        meta = json.loads((d / "meta.json").read_text())
    return {
        "name": strip_prefix(name),
        "path": str(d),
        "config": config,
        "forward_source": forward_src,
        "spec_yaml": spec_yaml,
        "header": header,
        "meta": meta,
    }


def register(
    config_path: Optional[str] = None,
    forward_paths: Optional[List[str]] = None,
    hf_model_path: Optional[str] = None,
    no_repo_modeling: bool = False,
    name: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Copy user-provided files into the cache and auto-generate a spec.

    Two input modes:
      • `hf_model_path=DIR` — point at a local HuggingFace-format repo
        that bundles config.json + safetensors (+ optional modeling*.py).
        Auto-discovers everything and extracts per-tensor shape/dtype by
        reading safetensors headers, so shape validation Just Works.
      • `config_path=FILE, forward_paths=[FILE, …]` — weights-less
        registration for prototype / research models without a full
        checkpoint handy. Skips shape validation (no safetensors).

    Forward-source policy when `hf_model_path` is set:
      • default — repo's `modeling_*.py` + any `forward_paths` (additive)
      • `no_repo_modeling=True` — skip repo's modeling entirely; use only
        `forward_paths`. Useful when the repo ships a broken or outdated
        modeling file and you want to point at a fixed version elsewhere.

    Returns a dict {name, path, spec_yaml, status} describing the cached
    entry.
    """
    # Resolve name
    if name is None:
        name = _next_auto_name()
    name = _safe_name(name)
    dest = model_path(name)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{name} already exists at {dest}; pass --overwrite to replace")

    # HF-format repo path — auto-discover config + safetensors + modeling.
    # `forward_paths` semantics (controlled by no_repo_modeling):
    #   • no_repo_modeling=False (default): repo's modeling_*.py + any
    #     forward_paths (additive, de-duped by absolute path). Useful
    #     when the repo references external helpers pulled in via
    #     trust_remote_code.
    #   • no_repo_modeling=True: skip repo's modeling entirely; use only
    #     forward_paths. Escape hatch for a broken repo modeling file.
    hf_repo = None
    if hf_model_path:
        hf_repo = Path(hf_model_path)
        if not hf_repo.is_dir():
            raise FileNotFoundError(f"hf_model_path is not a directory: {hf_model_path}")
        if config_path is None:
            candidate = hf_repo / "config.json"
            if not candidate.is_file():
                raise FileNotFoundError(f"config.json not found under {hf_model_path}")
            config_path = str(candidate)

        if no_repo_modeling:
            combined = list(forward_paths or [])
        else:
            auto_fw = sorted(hf_repo.glob("modeling_*.py"))
            if not auto_fw:
                auto_fw = sorted(p for p in hf_repo.glob("*.py")
                                 if "modeling" in p.name.lower())
            combined = [str(p) for p in auto_fw]
            seen = {str(Path(p).resolve()) for p in combined}
            for extra in forward_paths or []:
                resolved = str(Path(extra).resolve())
                if resolved not in seen:
                    combined.append(extra)
                    seen.add(resolved)
        forward_paths = combined

    if config_path is None:
        raise ValueError("register() requires either hf_model_path or config_path")

    # config
    cfg_src = Path(config_path)
    if not cfg_src.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    config = json.loads(cfg_src.read_text())

    # forward sources (concat with file banners). They are the model's only
    # modeling source: spec generation refuses to run without them, so an
    # empty forward is rejected here, before anything is written.
    forward_parts = []
    for p in (forward_paths or []):
        src = Path(p)
        if not src.is_file():
            raise FileNotFoundError(f"forward source not found: {p}")
        forward_parts.append((src.name, src.read_text()))
    if not any(text.strip() for _, text in forward_parts):
        raise ValueError(
            f"Custom model 'custom:{name}' has no modeling source (no forward "
            f"files given, none found under hf_model_path, or all empty). "
            f"Register it with its modeling files (--forward / modeling_*.py).")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "config.json").write_text(json.dumps(config, indent=2))
    (dest / "forward.py").write_text("\n\n".join(
        f"# ======== {fname} ========\n{text}" for fname, text in forward_parts))

    # Tensor metadata: only `hf_model_path` produces it (via
    # `hf_metadata.scan_local_hf_repo`). Without it, shape validation is
    # skipped — caller can still register a config-only drop, just won't
    # get per-tensor diff in validate.
    patterns_meta: Optional[Dict[str, Any]] = None
    if hf_repo is not None:
        try:
            from backend.hf.metadata import metadata_record, scan_local_hf_repo
            # Registration is spec generation: the repo's own config names
            # the scheme its tensors are classified under.
            repo_config_path = Path(hf_repo) / "config.json"
            repo_config = (json.loads(repo_config_path.read_text())
                           if repo_config_path.is_file() else {})
            scheme, num_bits, unknown_method = resolve_scheme(repo_config)
            patterns_meta = scan_local_hf_repo(
                str(hf_repo), scheme=scheme, num_bits=num_bits,
                unknown_method=unknown_method)
            (dest / "checkpoint_header.json").write_text(
                json.dumps(metadata_record(patterns_meta), indent=2))
        except (OSError, ValueError) as e:
            # Non-fatal: the repo may be a config-only drop. Record the
            # failure in meta so the user knows why shape validation is off.
            patterns_meta = {"error": f"{type(e).__name__}: {e}"}

    # meta
    (dest / "meta.json").write_text(json.dumps({
        "name": name,
        "created_at": int(time.time()),
        "source_config": str(cfg_src.resolve()),
        "source_forwards": [str(Path(p).resolve()) for p in (forward_paths or [])],
        "source_hf_repo": str(hf_repo.resolve()) if hf_repo else None,
        "scan_error": patterns_meta.get("error") if isinstance(patterns_meta, dict) and "error" in patterns_meta else None,
    }, indent=2))

    # Write a ready-to-use spec-generation prompt. We don't auto-invoke any
    # agent — the user picks which coding agent (Claude / GPT / Cursor) to
    # feed the prompt to, and saves the resulting YAML as spec.yaml in this
    # dir. This keeps the CLI offline-friendly and agent-agnostic.
    status = {"prompt_written": False, "error": None}
    try:
        from backend.spec_worker import build_custom_prompt
        prompt = build_custom_prompt(name)
        (dest / "prompt.md").write_text(prompt)
        status["prompt_written"] = True
    except (ImportError, OSError, ValueError) as e:
        status["error"] = f"{type(e).__name__}: {e}"

    return {
        "name": name,
        "path": str(dest),
        "status": status,
    }


def unregister(name: str) -> bool:
    d = model_path(name)
    if d.is_dir():
        shutil.rmtree(d)
        return True
    return False


def run_spec_generation(name: str, agent_cli: Optional[str] = None) -> int:
    """Run the configured coding agent to produce spec.yaml for a
    registered custom model.

    Streams the agent's SSE events to stdout (same ones the frontend
    consumes over HTTP). Returns the process exit code: 0 when the
    agent reported `done.success=true` and a valid spec.yaml landed
    in the cache dir, 1 otherwise.

    This is the CLI-side equivalent of the `/generate-spec` endpoint;
    it reuses `spec_worker.stream_agent` + `prepare_custom_generate_prompt`
    so behavior, logging and prompt content stay in lockstep with the
    frontend path.
    """
    import shutil as _shutil
    import json as _json
    from backend import spec_worker

    if agent_cli:
        spec_worker.configure_runtime(agent_cli=agent_cli)

    try:
        binary = spec_worker.get_agent_binary()
    except ValueError as e:
        print(f"auto-generate: invalid agent configuration: {e}")
        return 1
    if not _shutil.which(binary):
        print(f"auto-generate: agent CLI {binary!r} not found on PATH")
        print("  install it, or pick another agent via --agent-cli / SPEC_AGENT_CLI")
        return 1

    model_id = f"{NAME_PREFIX}{name}"
    status = spec_worker.acquire(model_id)
    if status != "ok":
        print(f"auto-generate: spec worker busy ({status})")
        return 1

    try:
        # Registering a model starts its generation afresh.
        target_root = spec_worker.allocate_run_overlay(
            model_id, kind="generate", discard_draft=True)
        prep_result = spec_worker.prepare_custom_generate_prompt(
            name, target_root=target_root)
    except Exception as e:
        spec_worker.release(model_id)
        print(f"auto-generate: metadata fetch failed: {e}")
        return 1
    success = False
    try:
        for sse in spec_worker.stream_agent(model_id, prep_result):
            # stream_agent yields framed SSE strings ("data: {...}\n\n").
            # Pull out the JSON payload and print its text field for
            # a clean terminal-friendly log.
            for line in sse.splitlines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = _json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                text = event.get("text") or event.get("message") or ""
                if etype == "done":
                    success = bool(event.get("success"))
                    msg = text or f"success={success}"
                    print(f"[done] {msg}")
                elif text:
                    print(f"[{etype}] {text}")
    except KeyboardInterrupt:
        print("\nauto-generate: interrupted by user")
        return 1
    return 0 if success else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="custom_model",
                                  description="Register local models for HuggingArch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("register", help="Register a new custom model")
    p_reg.add_argument("--hf-model-path",
                       help="path to a local HuggingFace-format repo "
                            "(config.json + safetensors + optional modeling*.py). "
                            "Preferred; enables full shape validation.")
    p_reg.add_argument("--config",
                       help="path to config.json. Only needed when --hf-model-path is absent "
                            "(weights-less flow); with --hf-model-path the repo's config.json "
                            "is picked up automatically.")
    p_reg.add_argument("--forward", nargs="+",
                       help="modeling_*.py source file(s). Required when --hf-model-path is absent; "
                            "when --hf-model-path is set these are ADDED on top of whatever "
                            "modeling*.py is already in the repo (for external helpers pulled "
                            "in by trust_remote_code)")
    p_reg.add_argument("--no-repo-modeling", action="store_true",
                       help="Skip auto-discovering modeling_*.py from --hf-model-path; "
                            "use only the files passed via --forward. Escape hatch when "
                            "the repo's bundled modeling is broken or outdated.")
    p_reg.add_argument("--name", help="short name (default: custom_model_N)")
    p_reg.add_argument("--overwrite", action="store_true")
    p_reg.add_argument("--auto-generate", action="store_true",
                       help="After registration, immediately run the configured coding agent "
                            "to generate spec.yaml. Requires the agent CLI on PATH. Pick the "
                            "agent via --agent-cli or SPEC_AGENT_CLI env var.")
    p_reg.add_argument("--agent-cli",
                       help="Override SPEC_AGENT_CLI for this invocation (only relevant with "
                            "--auto-generate). E.g. 'claude', 'codex', 'kimi'.")

    sub.add_parser("list", help="List registered custom models")

    p_rm = sub.add_parser("unregister", help="Remove a registered custom model")
    p_rm.add_argument("name")

    p_show = sub.add_parser("show", help="Print cached metadata for a model")
    p_show.add_argument("name")

    args = ap.parse_args(argv)

    if args.cmd == "list":
        for name in list_all():
            print(f"  custom:{name}")
        return 0

    if args.cmd == "unregister":
        if unregister(args.name):
            print(f"removed {args.name}")
            return 0
        print(f"not found: {args.name}")
        return 1

    if args.cmd == "show":
        try:
            d = load(args.name)
        except FileNotFoundError as e:
            print(e); return 1
        print(f"name:        {d['name']}")
        print(f"path:        {d['path']}")
        print(f"has spec:    {bool(d['spec_yaml'])}")
        print(f"has tensors: {bool(d['header'])}")
        print(f"meta:        {d['meta']}")
        return 0

    if args.cmd == "register":
        if not args.hf_model_path and not args.config:
            ap.error("either --hf-model-path or --config is required")
        if args.no_repo_modeling and not args.hf_model_path:
            ap.error("--no-repo-modeling only applies to --hf-model-path flow")
        if args.no_repo_modeling and not args.forward:
            ap.error("--no-repo-modeling requires --forward (otherwise no modeling files would be registered)")
        if args.agent_cli and not args.auto_generate:
            ap.error("--agent-cli only applies with --auto-generate")
        out = register(
            config_path=args.config,
            forward_paths=args.forward,
            hf_model_path=args.hf_model_path,
            no_repo_modeling=args.no_repo_modeling,
            name=args.name,
            overwrite=args.overwrite,
        )
        print(f"registered: custom:{out['name']}")
        print(f"  path:     {out['path']}")
        st = out["status"]
        if st.get("prompt_written"):
            print(f"  prompt:   {out['path']}/prompt.md")
        else:
            print(f"  prompt:   FAILED — {st['error']}")
            print(f"            write {out['path']}/spec.yaml by hand")
            return 1

        if args.auto_generate:
            print("")
            print(f"Auto-generating spec via coding agent...")
            return run_spec_generation(out["name"], agent_cli=args.agent_cli)

        print("")
        print(f"Next: feed {out['path']}/prompt.md to your coding agent, save the")
        print(f"resulting YAML as {out['path']}/spec.yaml. Then run:")
        print(f"  curl localhost:9100/analyze -d '{{\"model_id\":\"custom:{out['name']}\"}}'")
        print(f"(or pass --auto-generate next time to kick off the agent directly)")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
