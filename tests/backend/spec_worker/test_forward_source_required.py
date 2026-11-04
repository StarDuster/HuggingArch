"""No spec-gen agent starts without the model's real modeling source.

Generate / refine / review / fix prep obtains whole module files (installed
transformers modeling module, HF remote-code files, a custom model's
forward.py). When no channel yields one, prep raises `ForwardSourceUnavailable`
before any agent launches, and the job ends with that typed error: a failed
fetch says retry, a model with no source anywhere says retrying will not help.
No network: the Hub client is monkeypatched.
"""
from __future__ import annotations

import httpx2  # the hub's HTTP client: what its calls raise
import pytest
from huggingface_hub import errors as hf_errors

from backend.spec_worker._ground_truth import ForwardSourceUnavailable
from backend.spec_worker._prompts import tasks

EXCERPT = {"source": "model_info_cache",
           "files": [{"path": "/cache/org__m.json", "text": "# ==== X ===="}]}


def _meta(config=None, forward_source=EXCERPT):
    return {"_snapshot_data": {"config": config or {},
                               "forward_source": forward_source}}


def _hub(monkeypatch, *, listing=None, download=None):
    """Stand-in Hub: `listing` / `download` are a return value or an exception."""
    def _call(outcome):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    monkeypatch.setattr("huggingface_hub.list_repo_files",
                        lambda **k: _call(listing if listing is not None else []))
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **k: _call(download))


def _http_error(cls, status):
    resp = httpx2.Response(status, request=httpx2.Request("GET", "https://hf.co"))
    return cls("probe", response=resp)


@pytest.mark.parametrize("failure", [
    httpx2.ConnectError("offline"),
    hf_errors.LocalEntryNotFoundError("offline and not cached"),
    _http_error(hf_errors.HfHubHTTPError, 503),
])
def test_a_failed_fetch_stops_prep_as_retryable(monkeypatch, tmp_path, failure):
    # auto_map names the file, so the download itself is what fails; an
    # offline cache miss must not pass for "the repo has no modeling file".
    _hub(monkeypatch, listing=failure, download=failure)
    meta = _meta({"auto_map": {"AutoModel": "modeling_m.MModel"}})
    with pytest.raises(ForwardSourceUnavailable) as err:
        tasks._stage_source(str(tmp_path), "org/m", meta)
    assert err.value.retryable
    assert err.value.error_kind == "forward_source_unreachable"
    assert "retry" in str(err.value)
    assert not (tmp_path / "source").exists()


def test_a_listing_failure_stops_prep_as_retryable(monkeypatch, tmp_path):
    _hub(monkeypatch, listing=httpx2.ConnectError("offline"))
    with pytest.raises(ForwardSourceUnavailable) as err:
        tasks._stage_source(str(tmp_path), "org/m", _meta())
    assert err.value.retryable


def test_a_repo_without_modeling_code_is_not_retryable(monkeypatch, tmp_path):
    _hub(monkeypatch, listing=["config.json", "model.safetensors"])
    with pytest.raises(ForwardSourceUnavailable) as err:
        tasks._stage_source(str(tmp_path), "org/m", _meta({"model_type": "mystery"}))
    assert not err.value.retryable
    assert err.value.error_kind == "forward_source_missing"
    assert "will not help" in str(err.value)


def test_an_auto_map_file_the_repo_lacks_is_not_retryable(monkeypatch, tmp_path):
    _hub(monkeypatch, listing=["config.json"],
         download=_http_error(hf_errors.RemoteEntryNotFoundError, 404))
    meta = _meta({"auto_map": {"AutoModel": "modeling_m.MModel"}})
    with pytest.raises(ForwardSourceUnavailable) as err:
        tasks._stage_source(str(tmp_path), "org/m", meta)
    assert not err.value.retryable


@pytest.mark.parametrize("failure", [
    _http_error(hf_errors.GatedRepoError, 403),
    _http_error(hf_errors.RepositoryNotFoundError, 404),
    _http_error(hf_errors.RevisionNotFoundError, 404),
    _http_error(hf_errors.DisabledRepoError, 403),
    hf_errors.HFValidationError("bad repo id"),
    hf_errors.OfflineModeIsEnabled("HF_HUB_OFFLINE"),
])
def test_a_repo_that_cannot_be_read_as_is_is_not_retryable(monkeypatch, tmp_path, failure):
    _hub(monkeypatch, listing=failure)
    with pytest.raises(ForwardSourceUnavailable) as err:
        tasks._stage_source(str(tmp_path), "org/m", _meta())
    assert not err.value.retryable
    assert "will not help" in str(err.value)


INCOMPLETE = [{"path": "/hf/modeling_wrap.py",
               "text": "from .models.decoder import FooForCausalLM\n"}]


def test_captured_source_missing_its_decoder_is_refused(tmp_path):
    captured = {"source": "hf_modeling_files", "files": INCOMPLETE}
    with pytest.raises(ForwardSourceUnavailable) as err:
        tasks._stage_source(str(tmp_path), "org/m", _meta(forward_source=captured))
    assert not err.value.retryable
    assert "decoder" in str(err.value)
    assert not (tmp_path / "source").exists()


def test_fetched_source_missing_its_decoder_is_refused(monkeypatch, tmp_path):
    import backend.hf.forward_source as fsrc
    monkeypatch.setattr(fsrc, "_try_hf_modeling_files", lambda *_: INCOMPLETE)
    with pytest.raises(ForwardSourceUnavailable) as err:
        tasks._stage_source(str(tmp_path), "org/m", _meta())
    assert not err.value.retryable


@pytest.mark.parametrize("build", ["build_custom_prompt", "build_custom_refine_prompt"])
def test_a_custom_model_without_forward_source_is_refused(monkeypatch, build):
    import backend.custom_model as custom
    monkeypatch.setattr(custom, "load", lambda name: {
        "config": {"model_type": "x"}, "forward_source": "", "header": None})
    args = ("c",) if build == "build_custom_prompt" else ("c", 1, 2, -1)
    with pytest.raises(ForwardSourceUnavailable) as err:
        getattr(tasks, build)(*args)
    assert not err.value.retryable


def test_a_captured_module_file_needs_no_hub(monkeypatch, tmp_path):
    # Native architecture: the capture already holds the installed
    # transformers modeling file, so the Hub is never asked.
    _hub(monkeypatch, listing=httpx2.ConnectError("must not be called"))
    native = {"source": "local_transformers",
              "files": [{"path": "/site/modeling_m.py", "text": "class M: ..."}]}
    staged = tasks._stage_source(str(tmp_path), "org/m", _meta(forward_source=native))
    assert [p.rsplit("/", 1)[-1] for p in staged] == ["modeling_m.py"]
    assert (tmp_path / "source" / "modeling_m.py").read_text() == "class M: ..."


@pytest.mark.parametrize("model_type, module", [
    ("llama", "modeling_llama.py"),
    ("gemma3_text", "modeling_gemma3.py"),  # google/gemma-3-1b-it's config
])
def test_a_repo_without_code_reads_the_installed_transformers(
        monkeypatch, tmp_path, model_type, module):
    _hub(monkeypatch, listing=["config.json", "model.safetensors"])
    staged = tasks._stage_source(str(tmp_path), "org/m", _meta({"model_type": model_type}))
    assert module in [p.rsplit("/", 1)[-1] for p in staged]


@pytest.mark.parametrize("failure, retryable", [
    (httpx2.ConnectError("offline"), True),
    (hf_errors.OfflineModeIsEnabled("HF_HUB_OFFLINE"), False),
])
def test_a_native_model_type_whose_repo_cannot_be_read_is_not_stood_in(
        monkeypatch, tmp_path, failure, retryable):
    # `llama` is in the installed transformers, but the Hub has not said the
    # repo ships no code: the transformers module is not the repo's forward.
    _hub(monkeypatch, listing=failure)
    with pytest.raises(ForwardSourceUnavailable) as err:
        tasks._stage_source(str(tmp_path), "org/m", _meta({"model_type": "llama"}))
    assert err.value.retryable is retryable


def test_a_fetched_module_file_is_staged(monkeypatch, tmp_path):
    hub_file = tmp_path / "hub" / "modeling_m.py"
    hub_file.parent.mkdir()
    hub_file.write_text("class M: ...")
    _hub(monkeypatch, listing=["config.json", "modeling_m.py"], download=str(hub_file))
    overlay = tmp_path / "overlay"
    staged = tasks._stage_source(str(overlay), "org/m", _meta())
    assert staged == [str(overlay / "source" / "modeling_m.py")]


def test_without_an_overlay_the_files_are_listed_in_place(tmp_path):
    native = {"files": [{"path": "/site/modeling_m.py", "text": "class M: ..."}]}
    assert tasks._stage_source(None, "org/m", _meta(forward_source=native)) == [
        "/site/modeling_m.py"]


def test_generate_prep_raises_before_the_agent(monkeypatch):
    _hub(monkeypatch, listing=httpx2.ConnectError("offline"))
    monkeypatch.setattr(tasks, "_build_prompt_context",
                        lambda model_id: ("sys", "gt", _meta()))
    with pytest.raises(ForwardSourceUnavailable):
        tasks.prepare_generate_prompt("org/m", target_root=None)


@pytest.mark.parametrize("module, prep, seed", [
    ("reviewer", "prepare_review", "seed_review_overlay"),
    ("fixer", "prepare_fix", "seed_fix_overlay"),
])
def test_review_and_fix_prep_raise_before_validating(monkeypatch, tmp_path,
                                                      module, prep, seed):
    import importlib
    mod = importlib.import_module(f"backend.spec_worker.{module}")
    _hub(monkeypatch, listing=httpx2.ConnectError("offline"))
    monkeypatch.setattr(mod, seed, lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(mod, "_build_prompt_context",
                        lambda model_id: ("sys", "gt", _meta()))
    monkeypatch.setattr(mod, "_invoke_validate",
                        lambda *a, **k: pytest.fail("validated"))
    with pytest.raises(ForwardSourceUnavailable):
        getattr(mod, prep)("org/m", record_id="r", **{
            "reviewer": {"reviewer_id": "u"}, "fixer": {"fixer_id": "u"}}[module],
            bundle=None)


def test_the_job_ends_with_the_typed_error_and_frees_the_lock(monkeypatch):
    from backend.spec_worker import worker
    released = []
    monkeypatch.setattr(worker, "release", released.append)

    def prep():
        raise ForwardSourceUnavailable("could not fetch; retry", retryable=True)

    done = list(worker.generation_job("org/m", prep, run_key="u|org/m")())[-1]
    assert done == {"type": "done", "success": False,
                    "message": "could not fetch; retry",
                    "error_kind": "forward_source_unreachable", "retryable": True}
    assert released == ["u|org/m"]
