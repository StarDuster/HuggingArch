#!/usr/bin/env bash
# The Kernel Bench service in local development, for a supervisor (`ha`) to
# call (docs/measured_calibration.md §8.4). The HA API and the service run
# side by side; the service is the only process that opens the lake.
#
#   dev_bench_service.sh secret RUN_DIR   write a new shared secret into
#                                         RUN_DIR/bench_secret (0600) — once
#                                         per `up`
#   dev_bench_service.sh env RUN_DIR      print the exports the API, its
#                                         compute workers, the cache warm and
#                                         the service all need:
#                                         HUGGINGARCH_BENCH_URL,
#                                         HUGGINGARCH_BENCH_SECRET_FILE
#   dev_bench_service.sh run              exec the service (127.0.0.1,
#                                         HUGGINGARCH_BENCH_PORT, default
#                                         8100); run from the repo root with
#                                         the BENCH_LAKE_* settings (.env)
#   dev_bench_service.sh pid              the running service's pid, if any
#   dev_bench_service.sh health           exit 0 when /health answers
set -euo pipefail

PORT="${HUGGINGARCH_BENCH_PORT:-8100}"

case "${1:-}" in
  secret)
    dir="${2:?run dir}"
    mkdir -p "$dir"
    (umask 077 && .venv/bin/python -c 'import secrets; print(secrets.token_hex(32))' \
      > "$dir/bench_secret.tmp" && mv "$dir/bench_secret.tmp" "$dir/bench_secret")
    chmod 0600 "$dir/bench_secret" ;;
  env)
    dir="${2:?run dir}"
    echo "export HUGGINGARCH_BENCH_URL=http://127.0.0.1:$PORT"
    echo "export HUGGINGARCH_BENCH_SECRET_FILE=$dir/bench_secret" ;;
  run)
    exec .venv/bin/python -m backend.service.bench.server --port "$PORT" ;;
  pid)
    pgrep -f '^\.venv/bin/python -m backend\.service\.bench\.server( |$)' || true ;;
  health)
    secret="$(cat "${HUGGINGARCH_BENCH_SECRET_FILE:?}")"
    curl --noproxy '*' -sf -m 3 -H "X-HA-Internal: $secret" \
      "http://127.0.0.1:$PORT/health" > /dev/null ;;
  *) echo "usage: dev_bench_service.sh {secret RUN_DIR|env RUN_DIR|run|pid|health}" >&2
     exit 2 ;;
esac
