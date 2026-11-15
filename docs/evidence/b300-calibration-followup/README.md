# B300 calibration follow-up

See [REPORT.md](REPORT.md) for current scope, controlled estimates and remaining error.

Kimi-K3.1-Pro target-only decode, TP8/DCP8/EP2, batch1/8, 2K/16K inputs, output512, KV64. The current native plan has114 physical keys, of which five match the original nine-row dataset and109 are new. Unsupported keys remain explicitly uncovered. The latest AgentX comparison was cancelled by the user before calibration; no newer-version AgentX result is claimed.

The shared HuggingArch checkout is untouched. Code and evidence are stored in the separate codex/b300-calibration worktree/branch, with no PR. Legacy queue/controller scripts describe their original runs; they are not a daemon or an active monitor.
