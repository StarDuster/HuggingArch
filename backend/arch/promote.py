"""Promote a `components/drafts/<name>.yaml` to `components/library/`.

Usage:
    python -m backend.arch.promote <name> [--rename <new_name>]

What it does:
  1. Confirms `components/drafts/<name>.yaml` exists.
  2. `git mv`s it to `components/library/<new_name or name>.yaml`.
  3. Reloads the spec library and verifies the promoted component
     still loads cleanly (catches "passed through drafts/'s lazy gate
     but fails the global validator" cases — e.g. role=attention
     missing kv_label, ffn_dense missing parallelism.shard_symbols).

A failed verification leaves the file in its NEW location and prints
the validator error; fix in place and re-run with no further git
operations needed (the file is already where it belongs).

Renaming is supported because draft names tend to bake in a model
prefix that the README explicitly warns against
(`dsv4_compressor_no_overlap` → `token_pool_compressor`). Rename at
promotion is the only chance to do this without breaking existing
specs that reference the old name; after promotion the name is
public.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


_COMPONENTS_DIR = Path(__file__).resolve().parent / "components"
_DRAFTS_DIR = _COMPONENTS_DIR / "drafts"
_LIBRARY_DIR = _COMPONENTS_DIR / "library"


def _git(args: list[str]) -> None:
    """Run a git command and abort on non-zero exit."""
    result = subprocess.run(
        ["git", *args],
        cwd=_COMPONENTS_DIR.parent.parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def promote(name: str, new_name: str | None = None) -> None:
    src = _DRAFTS_DIR / f"{name}.yaml"
    if not src.is_file():
        raise SystemExit(
            f"no draft at {src.relative_to(_COMPONENTS_DIR.parent.parent)} — "
            f"check the name (no `.yaml` suffix needed)."
        )
    final_name = new_name or name
    dst = _LIBRARY_DIR / f"{final_name}.yaml"
    if dst.exists():
        raise SystemExit(
            f"library already has a {final_name!r} component "
            f"({dst.relative_to(_COMPONENTS_DIR.parent.parent)}) — pick a "
            f"different --rename target."
        )

    # Use `git mv` so history follows the file. If it's not git-tracked
    # yet (a brand-new draft never `git add`-ed), fall back to plain mv.
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(src)],
        cwd=_COMPONENTS_DIR.parent.parent.parent,
        capture_output=True, text=True, check=False,
    ).returncode == 0
    if tracked:
        _git(["mv", str(src), str(dst)])
        print(f"git mv {src.name} → library/{dst.name}")
    else:
        src.rename(dst)
        print(f"mv {src.name} → library/{dst.name}  (was untracked)")

    # If we renamed, rewrite the yaml's `name:` field to match. Without
    # this the loader treats the filename and the declared name as
    # different keys and the rename is half-applied.
    if new_name is not None and new_name != name:
        text = dst.read_text(encoding="utf-8")
        # Targeted replace on the first `name: <old>` at column 0.
        # Avoid yaml round-trip — destroys formatting / comments.
        import re
        new_text, n = re.subn(
            rf"^name:\s*{re.escape(name)}\s*$",
            f"name: {new_name}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if n == 0:
            print(
                f"WARN: could not find `name: {name}` at column 0 in "
                f"{dst.name}; you may need to update it by hand."
            )
        else:
            dst.write_text(new_text, encoding="utf-8")
            print(f"     rewrote name: {name} → {new_name}")

    # Verification: reload and confirm the promoted component is now in
    # _component_lib (i.e. passed library-tier validation).
    logging.basicConfig(level=logging.WARNING)
    from backend.arch.loader import SpecLoader
    loader = SpecLoader()
    if final_name in loader._component_lib:
        print(f"OK: {final_name} loaded as library component.")
    else:
        print(
            f"WARN: {final_name} did not enter _component_lib after promotion. "
            f"Check the load warnings above; the file is at "
            f"{dst.relative_to(_COMPONENTS_DIR.parent.parent)} and you can "
            f"edit it in place. Common causes: role=attention without "
            f"`kv_label`, role=ffn_dense without `parallelism.shard_symbols`."
        )
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("name", help="draft component basename (no .yaml suffix)")
    p.add_argument(
        "--rename",
        metavar="NEW",
        default=None,
        help="rename to NEW during promotion (also rewrites yaml's `name:`)",
    )
    args = p.parse_args()
    promote(args.name, args.rename)


if __name__ == "__main__":
    main()
