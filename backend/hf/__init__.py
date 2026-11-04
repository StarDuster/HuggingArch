"""HuggingFace integration — Layer 1 (I/O boundary).

  header.py          `SafetensorsHeader` — every tensor a checkpoint file
                     holds, by the name the file gives it.
  metadata.py        hub / local-repo header readers, the record a snapshot
                     stores, and the totals the header implies.
  forward_source.py  locate + read modeling_*.py from local transformers,
                     HF cache, snapshots, or direct repo download.
  model_info.py      subprocess that runs `transformers.AutoModel.from_config`
                     under init_empty_weights to capture print(model) +
                     forward source bodies.
  task_gate.py       task sanity check — the tasks the analyzer answers
                     for (text-generation / image-text-to-text), read
                     from the hub's transformers_info, and the precise
                     refusal for a repo that resolves outside them.
  snapshot.py        write spec snapshots (config + metadata + forward source
                     + model_str) to arch/snapshots/<owner>__<name>.json.

Imports are explicit per module — `from backend.hf.metadata import ...` —
no umbrella re-export. The package boundary is only for grouping; module
seams stay sharp.
"""
