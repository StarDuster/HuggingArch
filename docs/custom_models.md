# Custom Models (non-HuggingFace sources)

By default HuggingArch fetches `config.json` / safetensors / `modeling_*.py`
from HF Hub as ground truth. For models that **aren't on HF Hub** —
internal checkpoints, research prototypes, local forks — use the
`custom:<name>` protocol: register local files via one CLI command, then
type `custom:<name>` in the frontend, and every feature (Dashboard / Layer
Groups / Compare / DAG drill-down) works without code changes.

## Architectural constraints

- **No special form.** A registered custom model appears in the
  supported-models list with a `custom` badge, and `custom:<name>` can also be
  typed straight into the Model ID input; the routing is entirely backend-side.
- **Not committed to git.** All files land under
  `~/.huggingarch/custom_models/<name>/`; the repo never sees any custom
  model artifact.
- **Dynamic registration.** `SpecLoader.match()` refreshes its cache
  directory on every call — newly registered specs are picked up
  immediately, no server restart needed.

## Cache directory layout

```
~/.huggingarch/custom_models/<name>/
  config.json          # user-provided, copied verbatim
  forward.py           # user-provided modeling_*.py source(s) (concatenated)
  checkpoint_header.json # the local safetensors headers, verbatim, when available
  prompt.md            # auto-generated spec-gen prompt
  spec.yaml            # written by the coding agent (or hand-edited)
  meta.json            # registration timestamp, source file absolute paths
```

## Workflow

### 1. Register

```bash
python -m backend.custom_model register \
    --hf-model-path /abs/path/to/local_hf_repo \
    [--name my_model] [--overwrite] \
    [--forward extra_helper.py ...] [--no-repo-modeling] \
    [--auto-generate [--agent-cli codex]]
```

`--hf-model-path` is preferred: it reads that repository's `config.json`,
discovers `modeling_*.py`, scans safetensors headers into
`checkpoint_header.json`, writes `meta.json`, and generates `prompt.md`. Files
passed with `--forward` are added to the repo's own modeling files (for helpers
pulled in by `trust_remote_code`); `--no-repo-modeling` uses only those. A model
with no non-empty modeling source is refused — the agent models the forward, so
config alone is not enough. Without `--auto-generate` there is no `spec.yaml`
yet. For a weights-less prototype,
use the alternate mode:

```bash
python -m backend.custom_model register \
    --config /abs/path/to/config.json \
    --forward /abs/path/to/modeling_xxx.py [/abs/path/to/modeling_attn.py ...] \
    [--name my_model]
```

Without `--name`, registration chooses the next `custom_model_N` identity.

### 2. Generate the spec

**Option A: from the Inspector (recommended)**

Type `custom:my_model` in the Model ID input → click Analyze:

1. The Inspector calls `/analyze` → no spec matches.
2. It starts a spec-generation job (signed in: `/api/spec/generate`;
   single-user mode: `/generate-spec`) and polls its status; the panel shows
   the agent's progress, the same as for an HF model.
3. The backend sees the `custom:` prefix and assembles the cached config,
   forward source and tensors into the prompt
   (`prepare_custom_generate_prompt`).
4. The agent writes the spec into the run's own overlay directory
   (`models/custom:my_model.yaml`); a successful anonymous run also stores it
   as the cached `spec.yaml`. When both exist, the overlay is read first.
5. The spec goes through the same validation as any other; Analyze then
   renders every page.

**Option B: local manual run**

`prompt.md` already ends with an instruction to "Write to
`~/.huggingarch/.../spec.yaml` using the Write tool", so the agent writes
the file itself — no shell redirection needed:

```bash
# Kimi / Claude Code / Codex all work — they ship a Write tool
kimi   -p "$(cat ~/.huggingarch/custom_models/my_model/prompt.md)" --print --yolo
claude -p "$(cat ~/.huggingarch/custom_models/my_model/prompt.md)" --permission-mode acceptEdits
```

If you use an agent without a Write tool, you'll need stdout redirection
plus suppressing intermediate output:

```bash
# kimi's --quiet is equivalent to --print --output-format text --final-message-only
kimi -p "$(cat ~/.huggingarch/custom_models/my_model/prompt.md)" --quiet \
    > ~/.huggingarch/custom_models/my_model/spec.yaml
```

> ⚠️ Do **not** use `kimi -p ... --print > spec.yaml` — `--print` by
> default writes thinking, tool calls, and final result all to stdout, so
> the YAML gets polluted. You must add `--quiet` (or
> `--final-message-only`) to keep only the final assistant message.

After that, just Analyze `custom:my_model` from the frontend.

### 3. Analyze

Type `custom:my_model` in the frontend and Analyze. The pipeline is
identical to HF models:

| Step                          | HF model                          | Custom model                                       |
|-------------------------------|-----------------------------------|----------------------------------------------------|
| `config.json`                 | `hf_hub_download`                 | Read `~/.huggingarch/.../config.json`              |
| Tensor dtype                  | safetensors header                | Read `checkpoint_header.json` (absent in weights-less mode) |
| Spec match                    | `backend/arch/models/*.yaml` | Plus `~/.huggingarch/.../spec.yaml`                |
| Dashboard / Layer Groups / DAG | ✓                                | ✓                                                  |
| Compare (any two)             | ✓                                 | ✓ (`custom:a` vs `Qwen/...` works too)             |
| `unmatched_nodes` refine signal | ✓                               | Only with `--hf-model-path` metadata               |
| Storage / tensor coverage gates | ✓                               | Cannot pass without `--hf-model-path` metadata     |

## Graceful degradation (weights-less registration)

- `backend.hf.metadata.fetch_metadata("custom:x")` falls back to a cached
  `tensor_patterns.json` when there is no `checkpoint_header.json`, and returns
  `{}` when neither exists.
- The validator cannot satisfy its checkpoint-backed storage and tensor-coverage
  gates from that empty skeleton, so a weights-less registration does not yield
  a top-level `valid: true` result.
- Rendering has no checkpoint tensor index to attach per-node dtype or matching
  evidence.
- The Per-Layer Breakdown's Dtype column shows `unknown`; the Dashboard's
  "HF Checkpoint Size" disappears.
- **Once a loadable `spec.yaml` exists, core derivations (FLOPs / params, DAG,
  Compare) remain available** — they use the spec and config rather than tensor
  metadata. This does not waive the validator's checkpoint gates.

This is fine for research prototypes and early iteration. For production use,
register a local HF-format repository so the validator can compare storage and
tensor coverage.

## Other CLI commands

```bash
python -m backend.custom_model list
python -m backend.custom_model show my_model
python -m backend.custom_model unregister my_model
```

## Internal modules

- `backend/custom_id.py` — the `custom:` prefix, cache root, and path
- `backend/custom_model.py` — cache I/O and CLI
- `backend/spec_worker/_prompts/tasks.py::build_custom_prompt` — assembles the prompt from
  local ground truth, with the absolute `spec.yaml` target path embedded
- `backend/spec_worker/_prompts/tasks.py::prepare_custom_generate_prompt` — the lazy prompt
  function for `/generate-spec` (custom branch)
- `backend/spec_worker/_prompts/tasks.py::build_custom_refine_prompt` — the
  prompt for refining an existing custom spec
- `backend/api/routes/spec.py::generate_spec` — dispatches on the `custom:` prefix
- `backend/hf/metadata.py::fetch_config` / `fetch_metadata` + `backend/arch/loader.py::SpecLoader.match` (`custom_spec_file`) — read from the cache (and a run's overlay) when the model id starts with `custom:`

## The only difference from the HF flow

HF model specs live at `backend/arch/models/*.yaml` and are tracked
by git; custom-model specs live at
`~/.huggingarch/custom_models/.../spec.yaml` and are **never committed to
git**. The agent prompt has a different target path, but everything else is
identical — release builds need no special handling for custom models.
