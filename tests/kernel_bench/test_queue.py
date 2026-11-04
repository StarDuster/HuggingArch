"""The measuring loop's decisions, without a GPU.

What ran the H100 box for three days was a shell script on that machine's
local disk; these are the parts of it worth pinning — which model is next,
what counts as already measured, what a stop leaves behind, and when the
card needs holding.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernel_bench import keepalive
from kernel_bench.box import BoxProfileError, require_work_dir
from kernel_bench.queue import (Workspace, hand_over, measured_in_for,
                                model_of, pending, run_all_command,
                                run_once, serve, slug_of, unmeasured)


def _plan(space: Workspace, model_id: str, jobs: int = 1) -> Path:
    space.prepare()
    path = space.plan_of(model_id)
    path.write_text("".join('{"signature":"gemm"}\n' for _ in range(jobs)),
                    encoding="utf-8")
    return path


def _measured(space: Workspace, model_id: str, env: str = "torch") -> None:
    out = space.results_of(model_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{env}.csv").write_text("model,op\nx,y\n", encoding="utf-8")


class TestTheSlugIsReversible:
    def test_one_underscore_would_lose_the_name(self):
        # A single underscore is what the shell era used and it cannot be
        # undone: model names contain underscores.
        for model_id in ("google/gemma-2-9b", "nvidia/NVIDIA-Nemotron-3_Nano"):
            assert model_of(slug_of(model_id)) == model_id
        assert slug_of("google/gemma-2-9b") == "google__gemma-2-9b"


class TestWhatIsNext:
    def test_oldest_plan_first_and_done_is_skipped(self, tmp_path):
        space = Workspace(tmp_path).prepare()
        first = _plan(space, "org/first")
        second = _plan(space, "org/second")
        os.utime(first, (1_000, 1_000))
        os.utime(second, (2_000, 2_000))
        assert pending(space) == ["org/first", "org/second"]

        space.done_marker("org/first").parent.mkdir(parents=True, exist_ok=True)
        space.done_marker("org/first").touch()
        assert pending(space) == ["org/second"]

    def test_a_plan_handed_over_clears_the_marker(self, tmp_path):
        # Re-planning a model is how a fixed plan reaches the card; the
        # old `.done` would otherwise keep it measured-and-forgotten.
        space = Workspace(tmp_path).prepare()
        _plan(space, "org/m")
        space.done_marker("org/m").parent.mkdir(parents=True, exist_ok=True)
        space.done_marker("org/m").touch()
        staged = tmp_path / "staged.jsonl"
        staged.write_text('{"signature":"gemm"}\n', encoding="utf-8")
        hand_over(space, "org/m", staged)
        assert not space.done_marker("org/m").exists()
        assert pending(space) == ["org/m"]
        assert not staged.exists(), "the plan is moved in, not copied"

    def test_unmeasured_is_what_this_box_has_not_done(self, tmp_path):
        space = Workspace(tmp_path).prepare()
        space.done_marker("org/a").parent.mkdir(parents=True, exist_ok=True)
        space.done_marker("org/a").touch()
        assert unmeasured(space, ["org/a", "org/b"]) == ["org/b"]


class TestWhatCountsAsAlreadyMeasured:
    def test_every_other_model_counts_and_so_does_its_own(self, tmp_path):
        # The physical key carries no model name, so another model's cell
        # IS this model's cell. Its own results count too: a run resumed
        # after a crash must not re-measure what it already wrote.
        space = Workspace(tmp_path).prepare()
        _measured(space, "org/mine", "torch")
        _measured(space, "org/other", "vllm")
        got = measured_in_for(space, "org/mine", extra=["/exports/corpus.csv"])
        assert [Path(p).name for p in got] == ["torch.csv", "vllm.csv",
                                               "corpus.csv"]
        assert str(space.results_of("org/mine")) in got[0]

    def test_the_command_names_them_all(self, tmp_path):
        space = Workspace(tmp_path).prepare()
        _plan(space, "org/mine")
        _measured(space, "org/other")
        cmd = run_all_command(space, "org/mine", box="a-box", budget="2h")
        assert "run-all" in cmd and "--box" in cmd
        assert cmd[cmd.index("--budget") + 1] == "2h"
        assert cmd.count("--measured-in") == 1
        assert cmd[cmd.index("--jobs") + 1] == str(space.plan_of("org/mine"))


class TestRunningAndStopping:
    def test_a_measured_model_is_marked_and_logged(self, tmp_path):
        space = Workspace(tmp_path).prepare()
        _plan(space, "org/m", jobs=3)
        seen = []
        code = run_once(space, "org/m", spawn=lambda cmd: seen.append(cmd) or 0)
        assert code == 0 and len(seen) == 1
        assert space.done_marker("org/m").exists()
        log = space.log.read_text(encoding="utf-8")
        assert "start org/m (3 jobs)" in log and "end org/m rc=0" in log

    def test_a_run_that_did_not_finish_is_not_done(self, tmp_path):
        # Done means the run reached the end. A run that could not start is
        # a different thing: handing `run-all` a box NAME where a path
        # belongs refused all 61 models in a second, and marking each done
        # would have retired the whole corpus unmeasured.
        space = Workspace(tmp_path).prepare()
        _plan(space, "org/m")
        assert run_once(space, "org/m", spawn=lambda cmd: 2) == 2
        assert not space.done_marker("org/m").exists()

    def test_a_failing_model_does_not_hold_the_queue(self, tmp_path):
        # It keeps its place — a restart retries it — but the loop moves on
        # rather than spinning on the same failure while the card idles.
        space = Workspace(tmp_path).prepare()
        import os as _os
        first = _plan(space, "org/broken")
        second = _plan(space, "org/fine")
        _os.utime(first, (1_000, 1_000))
        _os.utime(second, (2_000, 2_000))
        ran = []

        def spawn(cmd):
            ran.append(cmd)
            return 1 if "org__broken" in " ".join(cmd) else 0

        serve(space, poll=0.01, once=False, spawn=spawn,
              stop_when_idle=True)
        assert len(ran) == 2, "the queue stopped at the broken model"
        assert space.done_marker("org/fine").exists()
        assert not space.done_marker("org/broken").exists()

    def test_a_model_stood_aside_gets_another_turn_next_pass(self, tmp_path,
                                                             monkeypatch):
        """`EXIT_PATHS_DISABLED` means "ask again", so the pass must roll over.

        `run_once` returns 4 for a run that REACHED THE END with a launch
        path switched off by its canary: those cells were never attempted, so
        the model keeps its place and must be asked again. The loop used to
        hold the set of stood-aside models for the whole process, which made
        a restart the only retry there was — and since most runs on the box
        end rc=4 by design, a queue 1 day old had every remaining model in
        that set and slept for three hours with 14 plans pending
        (2026-10-31). Nothing was down and nothing logged.

        `time.sleep` is replaced so the old behaviour ends the loop instead
        of hanging the suite: without it the old code idles forever and this
        test would time out rather than fail, which is a worse signal.
        """
        import kernel_bench.queue as q

        space = Workspace(tmp_path).prepare()
        _plan(space, "org/canary-off")
        ran, slept = [], []

        def spawn(cmd):
            ran.append(cmd)
            return 4                      # cli.EXIT_PATHS_DISABLED

        def fake_sleep(seconds):
            slept.append(seconds)
            if len(slept) > 2:
                raise q.Stopped("slept enough")

        monkeypatch.setattr(q.time, "sleep", fake_sleep)
        serve(space, poll=0, once=False, spawn=spawn)

        assert len(ran) >= 2, (
            "a model that ended rc=4 was asked once and never again: the "
            "stood-aside set outlived its pass")
        assert not space.done_marker("org/canary-off").exists(), (
            "rc=4 must not mark the model done — its cells were never run")

    def test_dry_run_measures_nothing(self, tmp_path, capsys):
        space = Workspace(tmp_path).prepare()
        _plan(space, "org/m")
        run_once(space, "org/m", dry_run=True,
                 spawn=lambda cmd: pytest.fail("spawned under --dry-run"))
        assert not space.done_marker("org/m").exists()
        assert "run-all" in capsys.readouterr().out

    def test_serve_drains_the_queue_and_leaves_no_pidfile(self, tmp_path):
        space = Workspace(tmp_path).prepare()
        _plan(space, "org/a")
        _plan(space, "org/b")
        ran = []
        while pending(space):
            serve(space, once=True, spawn=lambda cmd: ran.append(cmd) or 0)
        assert len(ran) == 2
        assert not space.pidfile.exists()


class TestAFloorStopsWhereItsOracleDoes:
    def test_the_scan_floors_cap_at_the_oracle_s_step_budget(self):
        """One fact, one number, two readers.

        The torch floors for the sequential-scan signatures RUN the oracle's
        own scan, so they cost what it costs: one Python step per token,
        each step launching a handful of small kernels. Measured on the
        H100's own queue log: 917 cells at 8.8 hours of single launches,
        48 seconds for one launch at 262,144 steps against about 8 ms of
        arithmetic — a figure that bounds nothing, on rows the oracle had
        already declined to check at the same step count.

        So the ceiling is the oracle's budget, read from it rather than
        restated. A second number here would drift from the one in
        `refs.py`, and then the floor would measure cells its own reference
        refuses.
        """
        from kernel_bench.providers import all_providers
        from kernel_bench.refs import SCAN_MAX_STEPS

        floors = {p.kernel_impl: p for p in all_providers()
                  if p.library == "torch" and p.kernel_impl in (
                      "torch_gated_delta_rule", "torch_kda",
                      "torch_mamba2_scan", "torch_gla_simple_recurrence")}
        assert len(floors) == 4, sorted(floors)
        for impl, provider in floors.items():
            assert provider.serves.max_dims == (("s_query", SCAN_MAX_STEPS),), (
                f"{impl} does not stop where its oracle does")

    def test_a_decode_cell_is_untouched_by_that_ceiling(self):
        """The ceiling is on STEPS, and a decode step is one.

        Capping these floors must not take the decode cells with them: B is
        in the thousands there and T is 1, which is one Python step — the
        case the oracle's own budget was wrongly refusing until it was
        spent on T instead of B*T.
        """
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import KernelSignature
        from kernel_bench.refs import SCAN_MAX_STEPS

        serves = KernelSignature(signature="gated_delta_rule",
                                 activation=("bf16",),
                                 max_dims=(("s_query", SCAN_MAX_STEPS),))
        def cell(tokens, batch):
            return Invocation.from_job({
                "signature": "gated_delta_rule", "op": "gated_delta_rule",
                "dims": {"B": batch, "s_query": tokens, "n_lin_k_heads": 4,
                         "n_lin_v_heads": 8, "d_lin_k_head": 128,
                         "d_lin_v_head": 128},
                "phase": "decode" if tokens == 1 else "prefill",
                "dtype_env": {"activation": "bf16", "weight": "bf16",
                              "kv": "bf16", "output": "bf16",
                              "metadata": "fp32"}})
        assert serves.accepts(cell(1, 4096)) is True
        assert serves.accepts(cell(256, 1)) is True
        assert serves.accepts(cell(262144, 1)) is False
        # The token says the clause AND the ceiling it crossed, which is
        # what makes a tally of refusals actionable.
        assert list(serves.refusals(cell(262144, 1))) == [
            "max_dims:s_query=262144 exceeds %d" % SCAN_MAX_STEPS]


class TestTheCardIsHeldOnlyWhenIdle:
    def test_the_burst_waits_for_the_limit_and_yields_to_real_work(self):
        assert not keepalive.should_burst(599, busy=False, idle_limit=600)
        assert keepalive.should_burst(600, busy=False, idle_limit=600)
        # Never against real work, however long the card looked idle.
        assert not keepalive.should_burst(6000, busy=True, idle_limit=600)

    def test_the_watch_writes_down_the_longest_unheld_stretch(self):
        """Not just the bursts it ran — the number the reclaim is measured
        against.

        A log of bursts cannot answer "how long was the card unheld", and
        reading the card from outside on a 5-minute grid cannot either: a
        run of empty samples is equally one long hole and a series of short
        gaps the sampler landed in. That ambiguity was read as a broken
        invariant on 2026-10-23 and it was not one, so the watch reports
        the stretch itself.
        """
        said: list = []
        samples = iter([
            frozenset(),            # the card goes quiet …
            frozenset(),
            frozenset({"7"}),       # … a measurement arrives before the limit
            frozenset({"7"}),
        ])

        def pids(run=None):
            try:
                return next(samples)
            except StopIteration:
                raise _Stop from None

        class _Stop(Exception):
            pass

        import kernel_bench.keepalive as ka
        real_pids, real_sleep = ka.compute_pids, ka.time.sleep
        ka.compute_pids = pids
        ka.time.sleep = lambda _s: None
        try:
            with pytest.raises(_Stop):
                ka.serve(poll=60.0, idle_limit=600.0, report_every=0.0,
                         log=said.append)
        finally:
            ka.compute_pids, ka.time.sleep = real_pids, real_sleep

        # No burst was due (120s of quiet against a 600s limit), and the
        # watch still said what the quiet was.
        assert said, "the watch reported nothing at all"
        assert any("longest unheld 120s" in line for line in said), said
        assert all("burst" not in line or "0 burst" in line for line in said)

    def test_a_burst_runs_in_a_child_so_its_context_ends_with_it(
            self, tmp_path, monkeypatch):
        """The keepalive's own process never touches the card: a CUDA
        context outlives its tensors, and the sweep refuses a card someone
        else has a context on."""
        out = tmp_path / "pid"
        monkeypatch.setenv("KEEPALIVE_TEST_PID_FILE", str(out))
        keepalive.hold(0.0, target=_record_pid)
        assert int(out.read_text()) != os.getpid()

    def test_an_unreadable_card_reports_nothing_rather_than_guessing(self):
        def explode(*_args, **_kwargs):
            raise OSError("no nvidia-smi here")
        assert keepalive.compute_pids(run=explode) == frozenset()


class TestTheWorkDirIsDeclared:
    def test_a_box_without_one_is_told_what_to_add(self, tmp_path):
        from kernel_bench.box import load_box_profile
        path = tmp_path / "box.toml"
        path.write_text('name = "nameless"\n', encoding="utf-8")
        with pytest.raises(BoxProfileError, match="work_dir"):
            require_work_dir(load_box_profile(path))

    def test_the_h100_box_takes_it_from_this_machine(self):
        """The profile names a hardware stack, so where the disk is mounted
        comes from the machine running it — no allocated machine's path in
        the repository, and one profile for every box with that stack."""
        from kernel_bench.box import load_box_profile
        root = Path(__file__).resolve().parents[2]
        profile = load_box_profile(
            root / "kernel_bench" / "envs" / "boxes" / "h100-sxm-cu128.toml")
        assert require_work_dir(
            profile, environ={"KERNEL_BENCH_WORK_DIR": "/mnt/scratch"}
        ) == Path("/mnt/scratch")

    def test_an_unset_work_dir_variable_is_refused_not_invented(self):
        """`expandvars` would leave `${KERNEL_BENCH_WORK_DIR}` as a literal
        and the queue would create a directory by that name, measuring into
        somewhere nobody meant."""
        from kernel_bench.box import load_box_profile
        root = Path(__file__).resolve().parents[2]
        profile = load_box_profile(
            root / "kernel_bench" / "envs" / "boxes" / "h100-sxm-cu128.toml")
        with pytest.raises(BoxProfileError, match="KERNEL_BENCH_WORK_DIR"):
            require_work_dir(profile, environ={})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def _record_pid(seconds, *, duty):
    """A stand-in burst for `hold`: records the pid it ran in."""
    Path(os.environ["KEEPALIVE_TEST_PID_FILE"]).write_text(str(os.getpid()))
