"""Per-user spec-gen overlay mechanics (Phase 1b/1c).

Covers the isolation primitives — overlay dir layout, the opt-in flag, the
overlay-scoped fresh-yaml finder, and the loader overlay kwargs — without a
full agent run. The end-to-end two-user isolation is exercised separately.
"""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest
from backend.spec_worker import overlay
from backend.spec_worker import worker as W
from backend.arch.loader import overlay_kwargs
from backend.spec_worker.bundle import BundleFile, FormalSource, normalize_bundle


def _seed(model_id: str, spec_yaml: str):
    return normalize_bundle(
        model_id, spec_yaml, [], FormalSource(models={}, components={}))


def test_peruser_flag(monkeypatch):
    """One rule reads HA_PERUSER_SPECGEN, and it is the one the routes use.

    `overlay` used to carry a second, narrower reading — no case folding, no
    yes/on, and False whenever the variable was unset — while the service
    route also treats any multi-user deployment as per-user by default. On a
    deployment with auth and no explicit flag the two disagreed, and nothing
    imported the overlay one.
    """
    from backend.service.routes.spec import _peruser_active

    monkeypatch.setattr("backend.supabase_env.jwt_verification_available", lambda: False)
    monkeypatch.delenv("HA_PERUSER_SPECGEN", raising=False)
    assert _peruser_active() is False
    for on in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("HA_PERUSER_SPECGEN", on)
        assert _peruser_active() is True, on
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("HA_PERUSER_SPECGEN", off)
        assert _peruser_active() is False, off

    # Unset on a multi-user deployment: per-user, without anyone saying so.
    monkeypatch.delenv("HA_PERUSER_SPECGEN", raising=False)
    monkeypatch.setattr("backend.supabase_env.jwt_verification_available", lambda: True)
    assert _peruser_active() is True


def test_new_run_overlay_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    ov = overlay.new_run_overlay("user-1", "meta/Foo-Bar")
    assert (ov / "models").is_dir()
    assert (ov / "components" / "drafts").is_dir()
    assert (ov / "snapshots").is_dir()
    assert (ov / "components" / "formal_proposals" / "library").is_dir()
    assert (ov / "components" / "formal_proposals" / "adapters").is_dir()
    assert (ov / ".agent-home").is_dir()
    assert (ov / ".tmp").is_dir()
    assert (ov / "backend" / "arch" / "models" / "qwen2.yaml").is_file()
    assert not (ov / "backend" / "arch" / "DESIGN.md").exists()
    reference = ov / "backend" / "arch" / "models" / "qwen2.yaml"
    assert stat.S_IMODE(reference.stat().st_mode) == 0o444
    # model id is path-safe (no nested dir from the slash), user-scoped
    assert ov.name == "generate__meta__Foo-Bar"
    assert ov.parent.name == "user-1"


def test_two_users_get_distinct_overlays(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    a = overlay.new_run_overlay("user-a", "meta/Foo")
    b = overlay.new_run_overlay("user-b", "meta/Foo")
    assert a != b  # same model, different users → isolated dirs


def test_generate_and_refine_overlays_are_distinct_leaves(tmp_path, monkeypatch):
    """Two in-flight jobs for the same user+model use different overlay roots;
    preparing generate does not delete refine's .agent-home."""
    from pathlib import Path
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    refine_p = Path(overlay.seed_run_overlay(
        "user-a", "meta/Foo", _seed("meta/Foo", "name: foo\n")))
    marker = refine_p / ".agent-home" / "keep"
    marker.write_text("live-refine")
    gen = overlay.new_run_overlay("user-a", "meta/Foo", kind="generate")
    assert gen.resolve() != refine_p.resolve()
    assert gen.name.startswith("generate__")
    assert refine_p.name.startswith("refine__")
    assert marker.read_text() == "live-refine"
    assert (refine_p / ".agent-home").is_dir()


def test_anonymous_generate_and_refine_use_distinct_leaves(tmp_path, monkeypatch):
    """Self-host / per-user-off uses the reserved __anonymous__ owner; generate
    and refine still do not share a write root (S4 §2.5)."""
    from pathlib import Path
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    gen = Path(overlay.allocate_run_overlay("meta/Foo", kind="generate"))
    ref = Path(overlay.allocate_run_overlay(
        "meta/Foo", kind="refine",
        seed_bundle=_seed("meta/Foo", "name: foo\n")))
    assert gen.parent.name == overlay.ANONYMOUS_OWNER
    assert ref.parent.name == overlay.ANONYMOUS_OWNER
    assert gen.resolve() != ref.resolve()
    assert gen.name.startswith("generate__")
    assert ref.name.startswith("refine__")
    marker = ref / ".agent-home" / "keep"
    marker.write_text("live-refine")
    overlay.allocate_run_overlay("meta/Foo", kind="generate")
    assert marker.read_text() == "live-refine"


def test_anonymous_custom_generate_and_refine_use_distinct_leaves(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    model_id = "custom:toy"
    gen = overlay.allocate_run_overlay(model_id, kind="generate")
    ref = overlay.allocate_run_overlay(
        model_id, kind="refine",
        seed_bundle=_seed(model_id, "match:\n  model_type: [toy]\n"))
    assert gen != ref
    assert overlay.is_anonymous_overlay(gen)
    assert overlay.is_anonymous_overlay(ref)


def test_run_overlay_reference_refresh_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    first = overlay.new_run_overlay("user-a", "meta/Foo")
    second = overlay.new_run_overlay("user-a", "meta/Foo")
    assert first == second
    reference = second / "backend" / "arch" / "models" / "qwen2.yaml"
    assert reference.is_file()
    assert stat.S_IMODE(reference.stat().st_mode) == 0o444


def test_run_overlay_copies_main_drafts_read_only(tmp_path, monkeypatch):
    """Merged drafts are reference inputs, not invisible main-only state."""
    source_arch = tmp_path / "source" / "backend" / "arch"
    for relative in (
        "models/qwen2.yaml",
        "components/library/pre_norm.yaml",
        "components/adapters/adapter.yaml",
        "components/drafts/merged_draft.yaml",
    ):
        path = source_arch / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("role: block\n", encoding="utf-8")
    (source_arch / "dims.yaml").write_text("symbols: {}\n", encoding="utf-8")
    monkeypatch.setattr(overlay, "__file__", str(
        tmp_path / "source" / "backend" / "spec_worker" / "overlay.py"))

    root = tmp_path / "overlay"
    root.mkdir()
    overlay._copy_agent_references(root)

    copied = root / "backend" / "arch" / "components" / "drafts" / "merged_draft.yaml"
    assert copied.read_text(encoding="utf-8") == "role: block\n"
    assert stat.S_IMODE(copied.stat().st_mode) == 0o444
    registry = root / "backend" / "arch" / "dims.yaml"
    assert registry.read_text(encoding="utf-8") == "symbols: {}\n"
    assert stat.S_IMODE(registry.stat().st_mode) == 0o444


def test_fresh_clears_all_authored_inputs_and_kinds_use_distinct_leaves(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    model = "meta/Foo"
    generate = overlay.new_run_overlay("user-a", model, kind="generate")
    for name in ("old-identity.yaml",):
        (generate / "models" / name).write_text("match: {}\n", encoding="utf-8")
    for relative in (
        "components/drafts/old.yaml",
        "components/formal_proposals/library/old.yaml",
    ):
        path = generate / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("role: attention\n", encoding="utf-8")
    (generate / "snapshots" / "old.json").write_text("{}", encoding="utf-8")

    generate_again = overlay.new_run_overlay("user-a", model, kind="generate")
    refine = overlay.new_run_overlay("user-a", model, kind="refine")

    assert generate_again == generate
    assert generate.name == "generate__meta__Foo"
    assert refine.name == "refine__meta__Foo"
    for leaf in (generate, refine):
        assert list((leaf / "models").glob("*")) == []
        assert list((leaf / "components" / "drafts").glob("*")) == []
        assert list((leaf / "snapshots").glob("*")) == []
        assert list((leaf / "components" / "formal_proposals").rglob("*.yaml")) == []
    assert (generate / ".agent-home").is_dir()
    assert (refine / ".agent-home").is_dir()


def test_seed_writes_serving_files_and_lazy_formal_proposals(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    proposal = BundleFile(
        "backend/arch/components/library/attention.yaml",
        "role: norm\ngraph: {out: {op: rmsnorm, in: x, dim: d_model}}\noutputs: [out]\n",
        formal_body_sha256="0" * 64,
    )
    seed = normalize_bundle(
        "meta/Foo", "match: {model_id: meta/Foo}\ndrafts: [new_attention]\n",
        [proposal], FormalSource(models={}, components={}),
        run_drafts={
            "new_attention": "role: norm\ngraph: {out: {op: rmsnorm, in: x, dim: d_model}}\noutputs: [out]\n",
        },
    )
    root = overlay.seed_run_overlay("user-a", "meta/Foo", seed)
    assert (Path(root) / "models" / "meta__Foo.yaml").is_file()
    assert (Path(root) / "components" / "drafts" / "new_attention.yaml").is_file()
    lazy = (Path(root) / "components" / "formal_proposals" / "library"
            / "attention.yaml")
    assert lazy.is_file()
    assert not (Path(root) / "components" / "library" / "attention.yaml").exists()


def test_fresh_model_yaml_scopes_to_overlay(tmp_path):
    (tmp_path / "models").mkdir()
    p = tmp_path / "models" / "foo.yaml"
    p.write_text("name: foo\n")
    got = W._fresh_model_yaml(time.time() - 5, root=str(tmp_path))
    assert got == p
    # empty overlay → None
    assert W._fresh_model_yaml(time.time() - 5, root=str(tmp_path / "empty")) is None


def test_private_retry_harvests_declared_drafts_even_when_old(tmp_path):
    drafts = tmp_path / "components" / "drafts"
    drafts.mkdir(parents=True)
    wanted = drafts / "wanted.yaml"
    wanted.write_text("name: wanted\n")
    (drafts / "unused.yaml").write_text("name: unused\n")
    old = time.time() - 3600
    os.utime(wanted, (old, old))

    harvested = W._harvest_component_yamls(
        [drafts], tmp_path, use_overlay=True,
        started_at=time.time(), declared_drafts={"wanted"},
    )

    assert harvested == [{
        "path": "backend/arch/components/drafts/wanted.yaml",
        "body": "name: wanted\n",
    }]


def test_overlay_kwargs():
    assert overlay_kwargs(None) == {}
    assert overlay_kwargs("") == {}
    kw = overlay_kwargs("/x")
    assert kw["overlay_model_dirs"] == ["/x/models"]
    assert kw["overlay_draft_dirs"] == ["/x/components/drafts"]


def test_stage_snapshot_makes_the_overlay_self_sufficient(tmp_path, monkeypatch):
    """The staged snapshot must round-trip through load_snapshot the way the
    agent's in-loop validate will read it (via $HA_SPEC_OVERLAY_DIR)."""
    monkeypatch.setenv("HA_SPEC_OVERLAY_DIR", str(tmp_path))
    data = {"config": {"model_type": "x"}, "metadata": {"total_params": 1},
            "forward_source": {"source": "hf_modeling_files",
                               "files": [{"path": "modeling_x.py",
                                          "text": "class X: ...\n"}]}}
    log = tmp_path / "run.log"

    W._stage_snapshot("acme/NoRepoSnapshot", data, str(tmp_path), str(log))

    from backend.arch.snapshots import load_snapshot
    snap = load_snapshot("acme/NoRepoSnapshot")
    assert snap is not None and snap["config"] == {"model_type": "x"}
    assert "staged for offline validate" in log.read_text()


def test_stage_snapshot_refuses_one_without_forward_source(tmp_path):
    """A capture with no source is not staged as a sourceless snapshot; the
    log says why."""
    log = tmp_path / "run.log"
    W._stage_snapshot("acme/NoRepoSnapshot",
                      {"config": {"model_type": "x"}, "metadata": {"t": 1}},
                      str(tmp_path), str(log))
    assert not (tmp_path / "snapshots").exists()
    assert "no forward source" in log.read_text()


def test_stage_snapshot_skips_when_there_is_nothing_to_stage(tmp_path):
    log = tmp_path / "run.log"
    W._stage_snapshot("acme/NoRepoSnapshot", None, str(tmp_path), str(log))
    W._stage_snapshot("acme/NoRepoSnapshot", {"config": {}}, str(tmp_path), str(log))
    W._stage_snapshot("custom:/x", {"config": {"a": 1}, "metadata": {"b": 2}},
                      str(tmp_path), str(log))
    W._stage_snapshot("acme/NoRepoSnapshot",
                      {"config": {"a": 1}, "metadata": {"b": 2}}, None, str(log))
    assert not (tmp_path / "snapshots").exists()


def test_resolve_run_workspace_returns_only_current_run(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    current = overlay.new_run_overlay("user-a", "meta/Foo")
    sibling = overlay.new_run_overlay("user-b", "meta/Foo")
    resolved = overlay.resolve_run_workspace(current)
    assert resolved == str(current.resolve())
    assert resolved != str(sibling.resolve())
    assert resolved != str((tmp_path / "runs").resolve())


def test_resolve_run_workspace_rejects_unmanaged_or_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    with pytest.raises(ValueError, match="requires one workspace"):
        overlay.resolve_run_workspace(None)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside managed overlay roots"):
        overlay.resolve_run_workspace(outside)
    with pytest.raises(ValueError, match="SpecGen workspace"):
        overlay.resolve_run_workspace(tmp_path / "runs")

    link = tmp_path / "runs" / "link"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside managed overlay roots"):
        overlay.resolve_run_workspace(link)


def test_gc_prunes_oldest_over_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    root = tmp_path / "runs"
    # 5 leaves across 2 users; stamp mtimes oldest→newest.
    made = []
    for i in range(5):
        d = root / f"user{i % 2}" / f"model{i}"
        (d / "models").mkdir(parents=True)
        made.append(d)
    base = time.time()
    for i, d in enumerate(made):
        t = base - (100 - i)  # d0 oldest … d4 newest, all well past min_age
        os.utime(d / "models", (t, t))
        os.utime(d, (t, t))
    # Keep 2, no age protection → 3 oldest removed.
    removed = overlay.gc_overlays(root, max_entries=2, min_age_s=0)
    assert removed == 3
    assert not made[0].exists() and not made[1].exists() and not made[2].exists()
    assert made[3].exists() and made[4].exists()


def test_gc_protects_fresh_overlays(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    root = tmp_path / "runs"
    for i in range(4):
        (root / "u" / f"m{i}" / "models").mkdir(parents=True)
    # All just created (mtime≈now) → min_age guard keeps every one despite cap.
    assert overlay.gc_overlays(root, max_entries=1, min_age_s=3600) == 0
    assert len(list((root / "u").iterdir())) == 4


def test_custom_spec_file_prefers_overlay(tmp_path, monkeypatch):
    from backend.arch.loader import custom_spec_file
    overlay_dir = tmp_path / "ov"
    (overlay_dir / "models").mkdir(parents=True)
    identity = overlay_dir / "models" / "custom:toy.yaml"
    identity.write_text("match:\n  model_type: [toy]\n")
    cache = tmp_path / "cache" / "toy"
    cache.mkdir(parents=True)
    (cache / "spec.yaml").write_text("match:\n  model_type: [stale]\n")
    monkeypatch.setattr("backend.arch.loader.model_path", lambda name: cache)
    assert custom_spec_file("custom:toy", str(overlay_dir)) == str(identity)
    assert custom_spec_file("custom:toy") == str(cache / "spec.yaml")
    assert custom_spec_file("meta/Foo", str(overlay_dir)) is None


def test_custom_prompts_echo_distinct_overlay_write_paths(tmp_path, monkeypatch):
    from backend.spec_worker._prompts import tasks
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr("backend.custom_model.load", lambda name: {
        "config": {"model_type": "toy"},
        "forward_source": "def forward(self, x):\n    return x\n",
        "header": None,
    })
    gen = overlay.allocate_run_overlay("custom:toy", kind="generate")
    ref = overlay.allocate_run_overlay(
        "custom:toy", kind="refine",
        seed_bundle=_seed("custom:toy", "match:\n  model_type: [toy]\n"))
    g = tasks.prepare_custom_generate_prompt("toy", target_root=gen)
    r = tasks.prepare_custom_refine_prompt("toy", 1, 2, 1, target_root=ref)
    assert g["target_root"] == gen and r["target_root"] == ref
    assert g["target_root"] != r["target_root"]
    assert f"{gen}/models/custom:toy.yaml" in g["prompt"]
    assert f"{ref}/models/custom:toy.yaml" in r["prompt"]
    from backend.custom_id import model_path as _custom_path
    cache_spec = str(_custom_path("toy") / "spec.yaml")
    assert cache_spec not in g["prompt"]
    assert cache_spec not in r["prompt"]


def test_land_anonymous_custom_writes_cache_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    cache = tmp_path / "custom_models" / "toy"
    monkeypatch.setattr(overlay, "custom_model_path", lambda name: cache)
    ov = overlay.allocate_run_overlay("custom:toy", kind="generate")
    body = "match:\n  model_type: [toy]\n"
    overlay.land_anonymous_overlay("custom:toy", ov, body)
    assert (cache / "spec.yaml").read_text() == body


def test_run_overlay_owner_rejects_reserved_anonymous_literal():
    with pytest.raises(ValueError, match="reserved overlay owner"):
        overlay.run_overlay_owner(overlay.ANONYMOUS_OWNER)
    assert overlay.run_overlay_owner("") == overlay.ANONYMOUS_OWNER
    assert overlay.run_overlay_owner("user-a") == "user-a"


def test_land_same_target_is_serialized_and_complete(tmp_path, monkeypatch):
    """Two lands of the same custom cache replace via temp+rename; the dest
    is always one complete body, never a mix of both writers."""
    import threading
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    cache = tmp_path / "custom_models" / "toy"
    monkeypatch.setattr(overlay, "custom_model_path", lambda name: cache)
    ov = overlay.allocate_run_overlay("custom:toy", kind="generate")
    bodies = ["match:\n  model_type: [aaa]\n", "match:\n  model_type: [bbb]\n"]
    errors = []

    def land(body):
        try:
            overlay.land_anonymous_overlay("custom:toy", ov, body)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=land, args=(bodies[0],))
    t2 = threading.Thread(target=land, args=(bodies[1],))
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert errors == []
    got = (cache / "spec.yaml").read_text()
    assert got in bodies


def test_land_skips_private_user_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    ov = overlay.new_run_overlay("user-a", "meta/Foo")
    assert not overlay.is_anonymous_overlay(ov)
    overlay.land_anonymous_overlay("meta/Foo", ov, "name: x\n")
    assert not (ov / "landed").exists()


def test_gc_noop_under_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    root = tmp_path / "runs"
    (root / "u" / "m0" / "models").mkdir(parents=True)
    assert overlay.gc_overlays(root, max_entries=200, min_age_s=0) == 0


def test_gc_keeps_a_leaf_whose_agent_home_is_active(tmp_path, monkeypatch):
    """A long round that writes no YAML for hours still appends to its CLI
    transcript under `.agent-home`; that activity protects the leaf."""
    root = tmp_path / "runs"
    old = time.time() - 10 * 3600
    leaves = []
    for i in range(3):
        leaf = root / "u" / f"m{i}"
        (leaf / "models").mkdir(parents=True)
        leaves.append(leaf)
    transcript = leaves[0] / ".agent-home" / ".claude" / "projects" / "-x" / "s.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")
    for leaf in leaves:
        for p in (leaf / "models", leaf):
            os.utime(p, (old, old))
    for p in transcript.parents:
        if p == leaves[0]:
            break
        os.utime(p, (old, old))
    removed = overlay.gc_overlays(root, max_entries=1, min_age_s=3600)
    assert removed == 2
    assert leaves[0].exists()
    assert not leaves[1].exists() and not leaves[2].exists()


def test_gc_default_age_covers_the_lock_hold(tmp_path, monkeypatch):
    """By default a leaf idle for less than the spec lock TTL — a 5h run
    between two writes — is not pruned."""
    from backend.spec_worker.limits import LOCK_TTL_SEC
    monkeypatch.delenv("HA_SPEC_OVERLAY_MIN_AGE_S", raising=False)
    root = tmp_path / "runs"
    idle = time.time() - (LOCK_TTL_SEC - 600)
    for i in range(3):
        leaf = root / "u" / f"m{i}"
        (leaf / "models").mkdir(parents=True)
        for p in (leaf / "models", leaf):
            os.utime(p, (idle, idle))
    assert overlay.gc_overlays(root, max_entries=1) == 0
