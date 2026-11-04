"""The image's entrypoint (`scripts/entrypoint.sh`, docs/measured_calibration.md
§8.1): both processes start with one new 0600 secret, the API only once the
service answers; when either exits, the other is stopped and reaped and the
entrypoint exits with the first one's status (fail-fast: Fly restarts the
machine); once the lake serves, the cache warm runs once, outside both.
`python` is a stand-in that records what it was started with and how it
ended, and answers the health probe as the test says."""
from __future__ import annotations

import os
import pathlib
import stat
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]

STAND_IN = r"""#!/usr/bin/env bash
# python -c ...: the secret generator, the real thing.
if [ "$1" = "-c" ]; then exec /usr/bin/env python3 "$@"; fi
# python - up|serving: the service's health probe.
if [ "$1" = "-" ]; then
  cat > /dev/null
  [ -f "$OUT/backend.service.bench.server" ] || exit 1
  [ "$2" = up ] && exit 0
  [ -n "$SERVING" ] && exit 0
  exit 1
fi
who="$2"                     # the module after `-m`
log="$OUT/$who"
echo "secret_file=$HUGGINGARCH_BENCH_SECRET_FILE url=$HUGGINGARCH_BENCH_URL args=${*:3}" >> "$log"
ls "$OUT" >> "$log"          # what had started before this one
exec > /dev/null 2>&1
# The two processes the entrypoint supervises run until stopped; every other
# module is a step that runs to completion, with the status the test sets in
# STATUS_<module, dots as underscores> (0 by default).
case "$who" in
  backend.api|backend.service.bench.server) ;;
  *) status="STATUS_${who//./_}"; exit "${!status:-0}" ;;
esac
if [ "$who" = "$EXIT_FIRST" ]; then sleep 1.5; exit "$EXIT_CODE"; fi
sleep 60 &
nap=$!
trap 'kill $nap; echo terminated >> "$log"; exit 143' TERM
wait $nap
"""


def _run(tmp_path, exit_first: str, code: int, *, serving: bool = True,
         warm: str = "on", pull: int = 0) -> tuple[int, float]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").write_text(STAND_IN)
    (bindir / "python").chmod(0o755)
    (bindir / "nice").write_text('#!/usr/bin/env bash\nshift 2\nexec "$@"\n')
    (bindir / "nice").chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "OUT": str(tmp_path),
           "EXIT_FIRST": exit_first, "EXIT_CODE": str(code),
           "SERVING": "1" if serving else "", "BENCH_SERVING_WAIT_S": "3",
           "HA_RUNTIME_WARM": warm, "STATUS_backend_sol_pool_artifact": str(pull),
           "HA_RUN_DIR": str(tmp_path / "run"), "HA_CACHE_ROOT": str(tmp_path / "cache")}
    t0 = time.monotonic()
    proc = subprocess.run(["bash", str(ROOT / "scripts" / "entrypoint.sh")], env=env,
                          capture_output=True, text=True, timeout=30, check=False)
    return proc.returncode, time.monotonic() - t0


def test_either_exit_stops_both_and_exits_with_its_status(tmp_path):
    for who, code in (("backend.api", 3), ("backend.service.bench.server", 5)):
        d = tmp_path / who
        d.mkdir()
        rc, took = _run(d, who, code)
        assert rc == code and took < 15
        other = ("backend.service.bench.server" if who == "backend.api" else "backend.api")
        assert (d / other).read_text().splitlines()[-1] == "terminated"


def test_one_secret_per_start_both_read_it(tmp_path):
    _run(tmp_path, "backend.api", 0)
    secret = tmp_path / "run" / "bench_secret"
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600 and len(secret.read_text().strip()) == 64
    for who in ("backend.api", "backend.service.bench.server"):
        assert (f"secret_file={secret} url=http://127.0.0.1:8100 args="
                in (tmp_path / who).read_text())


def test_the_api_starts_after_the_service_answers(tmp_path):
    _run(tmp_path, "backend.api", 0)
    started_before_api = (tmp_path / "backend.api").read_text().splitlines()[1:]
    assert "backend.service.bench.server" in started_before_api


def test_the_landing_warm_precedes_the_full_warm_after_the_api_opens(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _run(a, "backend.api", 0)
    warm_args = [line.split("args=", 1)[1]
                 for line in (a / "backend.warm_caches").read_text().splitlines()
                 if line.startswith("secret_file=")]
    assert warm_args == ["--landing", ""]
    started_before_api = (a / "backend.api").read_text().splitlines()[1:]
    assert "backend.service.bench.server" in started_before_api
    assert "backend.warm_caches" not in started_before_api
    _run(b, "backend.api", 0, serving=False)
    assert not (b / "backend.warm_caches").exists()


def test_the_runtime_warm_can_be_switched_off(tmp_path):
    _run(tmp_path, "backend.api", 0, warm="off")
    assert not (tmp_path / "backend.warm_caches").exists()


def test_a_failed_pool_bootstrap_starts_nothing(tmp_path):
    """A configured SOL pool artifact that cannot be pulled or verified stops
    the start with the pull's status, before either process runs: serving
    from a cold pool is what the bootstrap exists to prevent."""
    rc, took = _run(tmp_path, "backend.api", 0, pull=7)
    assert rc == 7 and took < 15
    assert not (tmp_path / "backend.api").exists()
    assert not (tmp_path / "backend.service.bench.server").exists()
