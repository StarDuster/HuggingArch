"""Back up a lake to the backup location now (backend/service/bench/backup.py),
and print the snapshot `current` names.

A lake built or repaired off the machine goes up here; the API on an empty
volume restores it at startup with HUGGINGARCH_BENCH_RESTORE=on
(docs/measured_calibration.md §4.7). The cutover used this path: the
one-time import from the Postgres tables (since dropped) ran offline.

Usage (the lake and the backup location from the environment, as the API
reads them; the API must not have the lake open):
  BENCH_LAKE_CATALOG=... BENCH_LAKE_DATA=... BENCH_LAKE_BACKUP_DATA=s3://bench-lake-backup/ \\
  BENCH_LAKE_BACKUP_S3_ENDPOINT=... BENCH_LAKE_BACKUP_S3_REGION=... \\
  BENCH_LAKE_BACKUP_S3_KEY=... BENCH_LAKE_BACKUP_S3_SECRET=... \\
    .venv/bin/python scripts/bench_lake_backup_now.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.service.bench.backup import Backup, BackupConfig
from backend.service.bench.lake import Lake, LakeConfig


def main() -> int:
    config = BackupConfig.from_env()
    if config is None:
        print("BENCH_LAKE_BACKUP_DATA is not set", file=sys.stderr)
        return 2
    lake = Lake.open(LakeConfig.from_env())
    try:
        backup = Backup(config.bucket())
        backup.run(lake)
        current = backup.current()
        print(f"backed up: snapshot {current['snapshot']} -> {config.root} ({current['manifest']})")
    finally:
        lake.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
