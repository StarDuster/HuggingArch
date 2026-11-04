"""Open a PR that contributes a generated spec to the library (Phase 5).

Writes a spec bundle to the repo via the GitHub REST API (no local git) and
opens a pull request. It NEVER merges — a human merges on GitHub after CI
passes, so a bad spec can't land unreviewed.

Auth — two modes, App preferred:

  * **GitHub App** (recommended): set ``HA_GITHUB_APP_ID`` +
    ``HA_GITHUB_APP_PRIVATE_KEY`` (PEM; single-line with literal ``\\n`` or
    base64-wrapped both work) and optionally ``HA_GITHUB_APP_INSTALLATION_ID``
    (auto-discovered from the repo when omitted). Branches/commits/PRs are
    attributed to the app's bot identity (``<app>[bot]``), so a real person's
    contribution isn't collapsed under the host operator's name. The actual
    contributor is credited via a ``Co-authored-by`` trailer + a PR-body line.
  * **Fine-grained PAT** (fallback): ``HA_GITHUB_TOKEN`` (Contents: write +
    Pull requests: write). Simpler, but every PR is opened under the token
    owner's personal account.

``HA_GITHUB_REPO`` defaults to ``shenh10/HuggingArch``. When neither auth mode
is configured, ``enabled()`` is False and callers surface a clear "not
configured" error rather than half-opening a PR.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
import tarfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory
from typing import Optional
from pathlib import Path

import httpx

from backend.arch.promote_op_proposal import (
    OPS_REGISTRY_PATH, PROPOSALS_DIR, REGISTRY_PATH, REPO_ROOT, read_artifacts,
    promote_spec_symbols, spliced_files,
)
from backend.arch.loader import SpecLoader, arch_fingerprint
from backend.arch.component_signature import component_structure_signature
from backend.spec_worker.bundle import (
    BundleFile,
    FormalComponent,
    FormalSource,
    SpecBundleError,
    normalize_bundle,
    rename_component_references,
)


logger = logging.getLogger(__name__)

_API = "https://api.github.com"
_API_VERSION = "2022-11-28"
_DEFAULT_REPO = "shenh10/HuggingArch"

# Installation-token cache: App tokens last ~1h; refresh proactively.
_inst_token: dict = {"token": None, "exp": 0.0}


class GitHubNotConfigured(RuntimeError):
    """No GitHub auth configured (neither App nor PAT) — contribute is off."""


def _app_id() -> Optional[str]:
    return os.environ.get("HA_GITHUB_APP_ID") or None


def _app_private_key() -> Optional[str]:
    """The App private key PEM. Accepts a raw PEM, a single-line PEM with
    literal ``\\n`` escapes (how Fly stores multi-line secrets), or a base64-
    wrapped PEM. Returns None when unset."""
    raw = os.environ.get("HA_GITHUB_APP_PRIVATE_KEY")
    if not raw:
        return None
    if "-----BEGIN" in raw:
        return raw.replace("\\n", "\n")
    try:  # maybe base64-wrapped
        dec = base64.b64decode(raw).decode("utf-8")
        if "-----BEGIN" in dec:
            return dec
    except (ValueError, UnicodeDecodeError):
        pass
    return raw


def _app_configured() -> bool:
    return bool(_app_id() and _app_private_key())


def enabled() -> bool:
    return _app_configured() or bool(os.environ.get("HA_GITHUB_TOKEN"))


def _repo() -> str:
    return os.environ.get("HA_GITHUB_REPO", _DEFAULT_REPO)


def _app_jwt() -> str:
    """Short-lived RS256 JWT identifying the App (max 10 min per GitHub; we use
    9, with a 60s backdate to tolerate clock skew)."""
    import jwt
    now = int(time.time())
    return jwt.encode(
        {"iat": now - 60, "exp": now + 540, "iss": _app_id()},
        _app_private_key(), algorithm="RS256",
    )


def _mint_installation_token() -> str:
    """Exchange the App JWT for an installation access token scoped to the repo.
    Discovers the installation id from the repo when not pinned via env."""
    jwt_headers = {
        "Authorization": f"Bearer {_app_jwt()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": _API_VERSION,
    }
    with httpx.Client(timeout=30.0, headers=jwt_headers) as cx:
        inst = os.environ.get("HA_GITHUB_APP_INSTALLATION_ID")
        if not inst:
            r = cx.get(f"{_API}/repos/{_repo()}/installation")
            r.raise_for_status()
            inst = str(r.json()["id"])
        r = cx.post(f"{_API}/app/installations/{inst}/access_tokens")
        r.raise_for_status()
        return r.json()["token"]


def _token() -> str:
    """The bearer token for REST calls: a cached App installation token when the
    App is configured, else the PAT. Raises when neither is set."""
    if _app_configured():
        now = time.time()
        if not _inst_token["token"] or now >= _inst_token["exp"]:
            _inst_token["token"] = _mint_installation_token()
            _inst_token["exp"] = now + 55 * 60  # refresh before the 1h expiry
        return _inst_token["token"]
    tok = os.environ.get("HA_GITHUB_TOKEN")
    if not tok:
        raise GitHubNotConfigured(
            "neither HA_GITHUB_APP_* nor HA_GITHUB_TOKEN is set")
    return tok


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": _API_VERSION,
    }


def _safe(model_id: str) -> str:
    """model_id → filename stem. Matches the snapshot convention (`/`→`__`);
    the filename is cosmetic since the loader matches specs by config."""
    return model_id.replace("/", "__")


_PROPOSAL_PREFIX = PROPOSALS_DIR.relative_to(REPO_ROOT).as_posix() + "/"
_OPS_REGISTRY_REL = str(OPS_REGISTRY_PATH.relative_to(REPO_ROOT))


def _splice_proposal_package(parts: dict) -> dict:
    """Canonical-file texts from an in-memory proposal package (patch.py, …)."""
    import tempfile
    from pathlib import Path
    if "patch.py" not in parts:
        raise ValueError(
            "op proposal package has no patch.py; cannot splice canonical files"
        )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name, body in parts.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"unsafe proposal path {name!r}")
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
        return spliced_files(read_artifacts(root))


def _bundle_files(model_id: str, spec_yaml: str,
                  new_component_yamls: Optional[list], snapshot_json: Optional[dict],
                  registry: str = "") -> dict:
    """repo-relative path → file text. spec + drafts + snapshot + op proposal.

    Proposal packages stay as incremental files under ops_proposals/. Splice
    into ops_registry.py / dims.yaml happens here, at Open PR, not in
    the generate job. So does the promotion of the spec's own `dim_symbols`
    into `registry` (main's dims.yaml): the library model binds rows of
    dims.yaml and declares none of its own.
    """
    safe = _safe(model_id)
    files: dict[str, str] = {}
    packages: dict[str, dict] = {}
    for comp in (new_component_yamls or []):
        path, body = comp.get("path"), comp.get("body")
        if not path or body is None:
            continue
        files[path] = body
        if path.startswith(_PROPOSAL_PREFIX) and "/_promoted/" not in path:
            rest = path[len(_PROPOSAL_PREFIX):]
            op_name, _, name = rest.partition("/")
            if op_name and name:
                packages.setdefault(op_name, {})[name] = body
    for parts in packages.values():
        files.update(_splice_proposal_package(parts))
    # On top of an op proposal's own dims.yaml rows when it has them.
    registry_rel = str(REGISTRY_PATH.relative_to(REPO_ROOT))
    try:
        promoted = promote_spec_symbols(
            model_id, spec_yaml, files.get(registry_rel, registry))
    except ValueError as exc:
        raise SpecBundleError(str(exc)) from exc
    files[f"backend/arch/models/{safe}.yaml"] = promoted.spec_yaml
    if promoted.registry is not None:
        files[registry_rel] = promoted.registry
    if snapshot_json:
        files[f"backend/arch/snapshots/{safe}.json"] = json.dumps(
            snapshot_json, indent=2, ensure_ascii=False
        )
    return files


def _coauthor_trailer(contributor: Optional[str]) -> str:
    """A ``Co-authored-by:`` git trailer when ``contributor`` carries an
    ``<email>`` (so GitHub can attribute the commit), else empty. Accepts
    ``"Name <email>"`` or a bare ``"<email>"``/``email``."""
    if not contributor:
        return ""
    m = re.search(r"<([^>]+@[^>]+)>", contributor)
    if m:
        name = contributor[:m.start()].strip() or "HuggingArch contributor"
        return f"\n\nCo-authored-by: {name} <{m.group(1)}>"
    if re.fullmatch(r"[^@\s]+@[^@\s]+", contributor.strip()):
        return f"\n\nCo-authored-by: HuggingArch contributor <{contributor.strip()}>"
    return ""


def coauthor_for_handle(handle: Optional[str]) -> Optional[str]:
    """Turn a GitHub username into a ``name <email>`` string whose email is the
    user's GitHub **noreply** address, so a ``Co-authored-by`` trailer built
    from it links to their profile. Resolves the numeric id (the modern
    ``<id>+<user>@users.noreply.github.com`` form GitHub links) via the API,
    falling back to the plain noreply on any error. Returns None for an empty
    handle or when GitHub auth isn't configured."""
    h = (handle or "").strip().lstrip("@")
    if not h:
        return None
    email = f"{h}@users.noreply.github.com"
    if enabled():
        try:
            with httpx.Client(timeout=10.0, headers=_headers()) as cx:
                r = cx.get(f"{_API}/users/{h}")
            if r.status_code == 200:
                uid = r.json().get("id")
                if uid:
                    email = f"{uid}+{h}@users.noreply.github.com"
        except (httpx.RequestError, GitHubNotConfigured) as e:
            logger.warning(f"coauthor id-resolve failed for {h}: {e}")
    return f"{h} <{email}>"


def _one_commit_on_base(cx: httpx.Client, repo: str, *, base_sha: str,
                        files: dict, message: str) -> str:
    """One git commit whose parent is ``base_sha`` and whose tree is that
    commit's tree plus ``files``. Returns the new commit SHA.

    The Contents API (`PUT .../contents/{path}`) makes one commit per file,
    which is why a spec bundle used to show up as N commits on the PR. The
    Git Data API writes blobs + one tree + one commit instead.
    """
    base = cx.get(f"{_API}/repos/{repo}/git/commits/{base_sha}")
    base.raise_for_status()
    entries = []
    for path, content in files.items():
        blob = cx.post(
            f"{_API}/repos/{repo}/git/blobs",
            json={"content": content, "encoding": "utf-8"},
        )
        blob.raise_for_status()
        entries.append({
            "path": path, "mode": "100644", "type": "blob",
            "sha": blob.json()["sha"],
        })
    tree = cx.post(
        f"{_API}/repos/{repo}/git/trees",
        json={"base_tree": base.json()["tree"]["sha"], "tree": entries},
    )
    tree.raise_for_status()
    commit = cx.post(
        f"{_API}/repos/{repo}/git/commits",
        json={"message": message, "tree": tree.json()["sha"],
              "parents": [base_sha]},
    )
    commit.raise_for_status()
    return commit.json()["sha"]


class MainMoved(RuntimeError):
    """The default branch is no longer at the commit the PR was built on."""


class BranchTakenOver(RuntimeError):
    """The PR branch is not where this run last saw it: someone else pushed."""


@dataclass(frozen=True)
class PullRequest:
    """An opened (or updated) PR and the commit this run put at its head."""

    url: str
    number: Optional[int]
    node_id: Optional[str]
    head_sha: str
    draft: bool
    base_sha: str


def _branch_head(cx: httpx.Client, repo: str, branch: str) -> Optional[str]:
    r = cx.get(f"{_API}/repos/{repo}/git/ref/heads/{branch}")
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()["object"]["sha"]


def _default_branch(cx: httpx.Client, repo: str) -> str:
    info = cx.get(f"{_API}/repos/{repo}")
    info.raise_for_status()
    return info.json().get("default_branch", "main")


def _point_branch(cx: httpx.Client, repo: str, branch: str, sha: str,
                  expected_head: Optional[str] = None, must_create: bool = False) -> None:
    """Create ``branch`` at ``sha``, or force-move it there if it exists.

    ``must_create`` refuses a branch that already exists (``BranchTakenOver``):
    a caller that saw no branch has no claim on one that appeared since.

    With ``expected_head`` the move is a compare-and-swap as far as the REST
    API allows (ADMIT I3): the head is read right before the move and must
    still be ``expected_head``, and read again right after it and must be
    ``sha``. Either mismatch means someone else pushed; the run stops with
    ``BranchTakenOver`` and leaves their commit in place."""
    if expected_head is None:
        mk = cx.post(
            f"{_API}/repos/{repo}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": sha},
        )
        if mk.status_code in (200, 201):
            return
        if mk.status_code != 422:
            mk.raise_for_status()
        if must_create:
            raise BranchTakenOver(f"{branch} appeared while this run wrote {sha}")
    else:
        head = _branch_head(cx, repo, branch)
        if head != expected_head:
            raise BranchTakenOver(
                f"{branch} is at {head}, not {expected_head} as this run read it")
    upd = cx.patch(
        f"{_API}/repos/{repo}/git/refs/heads/{branch}",
        json={"sha": sha, "force": True},
    )
    upd.raise_for_status()
    if expected_head is not None:
        head = _branch_head(cx, repo, branch)
        if head != sha:
            raise BranchTakenOver(
                f"{branch} moved to {head} while this run wrote {sha}")


def _graphql(cx: httpx.Client, query: str, variables: dict) -> dict:
    r = cx.post(f"{_API}/graphql", json={"query": query, "variables": variables})
    r.raise_for_status()
    body = r.json() or {}
    if body.get("errors"):
        raise httpx.HTTPStatusError(
            f"GitHub GraphQL error: {body['errors']}", request=r.request, response=r)
    return body.get("data") or {}


def set_pr_draft(cx: httpx.Client, node_id: str, draft: bool) -> None:
    """Turn an open PR into a draft, or mark it ready for review. The REST
    API cannot change a PR's draft state; GraphQL can."""
    if draft:
        _graphql(cx, "mutation($id: ID!) { convertPullRequestToDraft(input: "
                     "{pullRequestId: $id}) { pullRequest { isDraft } } }", {"id": node_id})
    else:
        _graphql(cx, "mutation($id: ID!) { markPullRequestReadyForReview(input: "
                     "{pullRequestId: $id}) { pullRequest { isDraft } } }", {"id": node_id})


def comment_on_pr(cx: httpx.Client, repo: str, number: int, body: str) -> None:
    r = cx.post(f"{_API}/repos/{repo}/issues/{number}/comments", json={"body": body})
    r.raise_for_status()


def open_files_pr(
    *,
    branch: str,
    files: dict,
    title: str,
    body: str,
    commit_prefix: str = "pr",
    contributor: Optional[str] = None,
    base_sha: Optional[str] = None,
    draft: bool = False,
) -> str:
    """Write ``files`` as ONE commit on the default branch, open (or
    update) a PR, and return its html_url. Never merges. See
    ``publish_files_pr``."""
    return publish_files_pr(
        branch=branch, files=files, title=title, body=body,
        commit_prefix=commit_prefix, contributor=contributor,
        base_sha=base_sha, draft=draft).url


def publish_files_pr(
    *,
    branch: str,
    files: dict,
    title: str,
    body: str,
    commit_prefix: str = "pr",
    contributor: Optional[str] = None,
    base_sha: Optional[str] = None,
    draft: bool = False,
    expected_head: Optional[str] = None,
    must_create: bool = False,
) -> PullRequest:
    """Write ``files`` as ONE commit, point ``branch`` at it, open (or
    update) its PR. Never merges.

    The commit's parent is ``base_sha`` when given — the main commit the
    files were computed against — else the default branch's tip. With
    ``base_sha`` the default branch is read again before the branch moves,
    and ``MainMoved`` is raised when it is somewhere else: files computed
    against an older main are not published as if they were current.
    ``expected_head`` makes the branch move conditional and ``must_create``
    refuses an existing branch (``_point_branch``).

    A re-push (update-pr / rebase-pr) lands on the same PR: its title and
    body are rewritten and its draft state follows ``draft``. Used by
    contributed specs and by the op-writer chain.
    """
    if not files:
        raise ValueError("open_files_pr: no files to write")
    repo = _repo()
    headers = _headers()
    trailer = _coauthor_trailer(contributor)
    names = ", ".join(sorted(p.rsplit("/", 1)[-1] for p in files))
    message = f"{commit_prefix}: {names}{trailer}"

    with httpx.Client(timeout=60.0, headers=headers) as cx:
        base = _default_branch(cx, repo)

        ref = cx.get(f"{_API}/repos/{repo}/git/ref/heads/{base}")
        ref.raise_for_status()
        tip = ref.json()["object"]["sha"]
        parent = base_sha or tip
        if parent != tip:
            raise MainMoved(f"{base} is at {tip}, not {parent}")

        commit_sha = _one_commit_on_base(
            cx, repo, base_sha=parent, files=files, message=message)
        if base_sha is not None:
            again = cx.get(f"{_API}/repos/{repo}/git/ref/heads/{base}")
            again.raise_for_status()
            if again.json()["object"]["sha"] != base_sha:
                raise MainMoved(f"{base} moved past {base_sha} while the commit was written")
        _point_branch(cx, repo, branch, commit_sha, expected_head=expected_head,
                      must_create=must_create)

        pr = cx.post(
            f"{_API}/repos/{repo}/pulls",
            json={"title": title, "head": branch, "base": base,
                  "body": body, "maintainer_can_modify": True, "draft": draft},
        )
        if pr.status_code == 422:
            # A PR already exists for this head — the force-update above
            # already moved it onto latest main. Return that PR (re-push
            # after an agent-fix / rebase lands on the SAME PR). The branch
            # is deterministic per record, so there is at most one.
            owner = repo.split("/")[0]
            existing = cx.get(f"{_API}/repos/{repo}/pulls",
                              params={"head": f"{owner}:{branch}", "state": "open"})
            existing.raise_for_status()
            items = existing.json() or []
            if items:
                item = items[0]
                number = item.get("number")
                if number is not None:
                    upd = cx.patch(f"{_API}/repos/{repo}/pulls/{number}",
                                   json={"title": title, "body": body})
                    upd.raise_for_status()
                if item.get("node_id") and bool(item.get("draft")) != draft:
                    set_pr_draft(cx, item["node_id"], draft)
                return PullRequest(url=item["html_url"], number=number,
                                   node_id=item.get("node_id"), head_sha=commit_sha,
                                   draft=draft, base_sha=parent)
        pr.raise_for_status()
        made = pr.json()
        return PullRequest(url=made["html_url"], number=made.get("number"),
                           node_id=made.get("node_id"), head_sha=commit_sha,
                           draft=bool(made.get("draft", draft)), base_sha=parent)


@dataclass(frozen=True)
class RemoteTree:
    """One commit of the repository: its SHA, its whole archive (the tree
    the admitter runs in) and the YAML namespace read out of it."""

    sha: str
    archive: bytes = field(repr=False)
    source: FormalSource = field(repr=False)


def remote_tree(cx: httpx.Client, repo: str, sha: Optional[str] = None) -> RemoteTree:
    """The commit ``sha``, or the default branch's tip, from one archive
    request."""
    if sha is None:
        branch = _default_branch(cx, repo)
        ref = cx.get(f"{_API}/repos/{repo}/git/ref/heads/{branch}")
        ref.raise_for_status()
        sha = ref.json()["object"]["sha"]
    archive = cx.get(
        f"{_API}/repos/{repo}/tarball/{sha}",
        headers={"Accept": "application/vnd.github+json"},
        follow_redirects=True,
    )
    archive.raise_for_status()
    return RemoteTree(sha=sha, archive=archive.content,
                      source=_formal_source_from_archive(archive.content))


def _remote_formal_source(cx: httpx.Client, repo: str) -> tuple[str, FormalSource]:
    """Read the default branch's YAML namespace from one archive request."""
    tree = remote_tree(cx, repo)
    return tree.sha, tree.source


def _formal_source_from_archive(content: bytes) -> FormalSource:
    models: dict[str, str] = {}
    components: dict[str, FormalComponent] = {}
    registry = ""
    tiers = {"library", "adapters", "drafts"}
    try:
        with tarfile.open(fileobj=BytesIO(content), mode="r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if not parts:
                    continue
                relative = PurePosixPath(*parts[1:])  # GitHub adds a repo-root dir
                rel_parts = relative.parts
                if relative.as_posix() == "backend/arch/dims.yaml":
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        registry = extracted.read().decode("utf-8")
                    continue
                if (len(rel_parts) == 4 and rel_parts[:3] == ("backend", "arch", "models")
                        and relative.suffix in (".yaml", ".yml")):
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        models[relative.stem] = extracted.read().decode("utf-8")
                    continue
                if (len(rel_parts) == 5 and rel_parts[:3] == ("backend", "arch", "components")
                        and rel_parts[3] in tiers and relative.suffix in (".yaml", ".yml")):
                    tier = rel_parts[3]
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    body = extracted.read().decode("utf-8")
                    path = relative.as_posix()
                    try:
                        signature = component_structure_signature(
                            body, path=path)
                    except (ValueError, TypeError, AttributeError) as exc:
                        raise SpecBundleError(
                            f"remote main component {path!r} is invalid: {exc}"
                        ) from exc
                    components[relative.stem] = FormalComponent(
                        tier=tier, body=body, signature=signature, path=path)
    except SpecBundleError:
        raise
    except (tarfile.TarError, OSError, UnicodeError) as exc:
        raise SpecBundleError(f"could not read GitHub main archive: {exc}") from exc
    return FormalSource(models=models, components=components,
                        registry=registry)


def _paginated_json(cx: httpx.Client, url: str, *, params: dict) -> list[dict]:
    """Read every page of a GitHub list endpoint (30 items is the default)."""
    items: list[dict] = []
    page = 1
    while True:
        response = cx.get(url, params={**params, "page": page, "per_page": 100})
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise SpecBundleError(f"GitHub returned a non-list response for {url}")
        items.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100:
            return items
        page += 1


def _head_file_body(cx: httpx.Client, repo: str, ref: str, path: str) -> Optional[str]:
    response = cx.get(
        f"{_API}/repos/{repo}/contents/{path}", params={"ref": ref})
    if response.status_code >= 400:
        return None
    value = response.json()
    if not isinstance(value, dict) or value.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(value.get("content") or "").decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        logger.warning("could not decode GitHub contents payload for %s: %s", path, exc)
        return None


def _reject_open_pr_draft_conflicts(
    cx: httpx.Client,
    repo: str,
    *,
    branch: str,
    drafts: list[BundleFile],
) -> None:
    """Refuse a same-name/different-body draft already pending in another PR."""
    candidate_paths = {file.path for file in drafts}
    if not candidate_paths:
        return
    pulls = _paginated_json(
        cx, f"{_API}/repos/{repo}/pulls", params={"state": "open"})
    for pull in pulls:
        head = pull.get("head") or {}
        if head.get("ref") == branch:
            continue
        changed = _paginated_json(
            cx,
            f"{_API}/repos/{repo}/pulls/{pull.get('number')}/files",
            params={},
        )
        overlap = [item for item in changed
                   if item.get("filename") in candidate_paths]
        for item in overlap:
            path = item["filename"]
            candidate = next(file for file in drafts if file.path == path)
            other = _head_file_body(
                cx, repo, str(head.get("ref") or ""), path)
            try:
                same = (
                    other is not None
                    and component_structure_signature(other, path=path)
                    == component_structure_signature(candidate.body, path=path)
                )
            except (ValueError, TypeError, AttributeError):
                same = False
            if not same:
                url = pull.get("html_url") or f"PR {pull.get('number')}"
                raise SpecBundleError(
                    f"{path} is already changed differently in open PR {url}. "
                    "Wait for that PR to merge or close, then update this PR; "
                    "or rename this draft and update its references.")


def preflight_remote_tree(
    *, branch: str, drafts: list[BundleFile]
) -> RemoteTree:
    """Fetch main and check open PR conflicts before any write request."""
    repo = _repo()
    with httpx.Client(timeout=60.0, headers=_headers()) as cx:
        tree = remote_tree(cx, repo)
        _reject_open_pr_draft_conflicts(
            cx, repo, branch=branch, drafts=drafts)
    return tree


def branch_head(branch: str) -> Optional[str]:
    """The commit ``branch`` points at, or None when it does not exist."""
    with httpx.Client(timeout=30.0, headers=_headers()) as cx:
        return _branch_head(cx, _repo(), branch)


def changed_paths(base: str, head: str) -> list[str]:
    """The paths ``head`` changes relative to its merge base with ``base``
    (every page of the compare API's file list)."""
    repo = _repo()
    with httpx.Client(timeout=30.0, headers=_headers()) as cx:
        paths: list[str] = []
        page = 1
        while True:
            r = cx.get(f"{_API}/repos/{repo}/compare/{base}...{head}",
                       params={"page": page, "per_page": 100})
            r.raise_for_status()
            batch = r.json().get("files") or []
            for item in batch:
                paths.append(item["filename"])
                if item.get("previous_filename"):
                    paths.append(item["previous_filename"])
            if len(batch) < 100:
                return sorted(set(paths))
            page += 1


def file_at(ref: str, path: str) -> Optional[str]:
    """The text of ``path`` at ``ref``, or None when it is not there."""
    with httpx.Client(timeout=30.0, headers=_headers()) as cx:
        return _head_file_body(cx, _repo(), ref, path)


def reconcile_pr(pr: "PullRequest", reason: str) -> None:
    """Turn ``pr`` into a draft and say why in a comment (ADMIT I4): what it
    published no longer stands on the review or the head it was built on."""
    repo = _repo()
    with httpx.Client(timeout=30.0, headers=_headers()) as cx:
        if pr.node_id and not pr.draft:
            set_pr_draft(cx, pr.node_id, True)
        if pr.number is not None:
            comment_on_pr(cx, repo, pr.number, reason)


def fetch_remote_tree(sha: Optional[str] = None) -> RemoteTree:
    """The commit ``sha`` (default: main's tip) with its archive."""
    with httpx.Client(timeout=60.0, headers=_headers()) as cx:
        return remote_tree(cx, _repo(), sha)


def _candidate_signature(file: BundleFile) -> str:
    try:
        return component_structure_signature(file.body, path=file.path)
    except (ValueError, TypeError, AttributeError) as exc:
        raise SpecBundleError(
            f"candidate draft {file.path!r} is invalid: {exc}") from exc


def _input_drafts(files: Optional[list]) -> list[BundleFile]:
    """PR-layout draft candidates before any formal-source normalization."""
    drafts: list[BundleFile] = []
    for item in files or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", ""))
        parts = PurePosixPath(path).parts
        if (len(parts) == 5 and parts[:4] == (
                "backend", "arch", "components", "drafts")
                and PurePosixPath(path).suffix in (".yaml", ".yml")):
            drafts.append(BundleFile(path=path, body=item.get("body", "")))
    return drafts


def _dedupe_remote_drafts(
    bundle, formal_source: FormalSource
):
    """Replace exact remote duplicates; return its bundle and rename map."""
    drafts = [file for file in bundle.serving_files
              if file.path.startswith("backend/arch/components/drafts/")]
    signatures = {
        file.path.rsplit("/", 1)[-1][:-5]: _candidate_signature(file)
        for file in drafts
    }

    by_signature: dict[str, list[FormalComponent]] = {}
    for component in formal_source.components.values():
        by_signature.setdefault(component.signature, []).append(component)
    tier_order = {"library": 0, "adapters": 1, "drafts": 2}
    for choices in by_signature.values():
        choices.sort(key=lambda item: (
            tier_order.get(item.tier, 99), item.path, _safe(item.path)))

    mapping: dict[str, str] = {}
    retained: list[BundleFile] = []
    for file in drafts:
        stem = file.path.rsplit("/", 1)[-1][:-5]
        matches = by_signature.get(signatures[stem], [])
        if matches:
            mapping[stem] = matches[0].path.rsplit("/", 1)[-1][:-5]
        else:
            retained.append(file)

    spec_yaml = bundle.spec_yaml
    serving_files: list[BundleFile] = []
    for file in bundle.serving_files:
        if file in drafts:
            continue
        body = (rename_component_references(file.body, mapping, label=file.path)
                if file.path.endswith((".yaml", ".yml")) else file.body)
        serving_files.append(BundleFile(
            path=file.path, body=body,
            formal_body_sha256=file.formal_body_sha256))
    serving_files.extend(BundleFile(
            path=file.path,
            body=rename_component_references(file.body, mapping, label=file.path),
        ) for file in retained)
    if mapping:
        spec_yaml = rename_component_references(
            spec_yaml, mapping, label="target spec")
    deduped = normalize_bundle(
        bundle.model_id,
        spec_yaml,
        [*serving_files, *bundle.proposal_files],
        formal_source,
    )
    return deduped, mapping


def _write_formal_preview(root: Path, source: FormalSource) -> None:
    for stem, body in source.models.items():
        path = root / "models" / f"{stem}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    for component in source.components.values():
        path = root / PurePosixPath(component.path).relative_to("backend/arch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(component.body, encoding="utf-8")


def _write_bundle_preview(root: Path, model_id: str, spec_yaml: str,
                          files: list[BundleFile]) -> Path:
    for file in files:
        pure = PurePosixPath(file.path)
        if pure.is_absolute() or ".." in pure.parts:
            raise SpecBundleError(f"unsafe PR preview path {file.path!r}")
        if pure.parts[:2] != ("backend", "arch"):
            continue
        path = root / PurePosixPath(*pure.parts[2:])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file.body, encoding="utf-8")
    target = root / "models" / f"{_safe(model_id)}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(spec_yaml, encoding="utf-8")
    return target


def _verify_remote_reuse(
    model_id: str,
    output: tuple,
    control_spec: str,
    control_drafts: list[BundleFile],
    formal_source: FormalSource,
) -> None:
    """Load both layouts locally and require identical architecture."""
    def load(loader: SpecLoader, target: Path, layout: str):
        try:
            arch = loader.load_model_spec(str(target))
        except Exception as exc:
            raise SpecBundleError(
                f"deduplicated PR {layout} spec could not be loaded against "
                "remote main; fix missing component references before "
                "opening the PR") from exc
        if arch is None:
            raise SpecBundleError(
                "deduplicated PR spec could not be loaded against remote main; "
                "fix missing component references before opening the PR")
        return arch

    output_spec, output_files = output
    with TemporaryDirectory(prefix="spec-pr-remote-") as td:
        output_root = Path(td) / "output" / "arch"
        control_root = Path(td) / "control" / "arch"
        _write_formal_preview(output_root, formal_source)
        _write_formal_preview(control_root, formal_source)
        output_target = _write_bundle_preview(
            output_root, model_id, output_spec, output_files)
        control_target = _write_bundle_preview(
            control_root, model_id, control_spec, control_drafts)
        output_arch = load(
            SpecLoader(base_dir=str(output_root)), output_target, "output")
        control_arch = load(
            SpecLoader(base_dir=str(control_root)), control_target, "control")
        try:
            output_fingerprint = arch_fingerprint(output_arch)
            control_fingerprint = arch_fingerprint(control_arch)
        except Exception as exc:
            raise SpecBundleError(
                "deduplicated PR spec architecture fingerprint failed against "
                "remote main; fix invalid component references before opening "
                "the PR") from exc
        if output_fingerprint != control_fingerprint:
            raise SpecBundleError(
                "deduplicated PR spec changes the architecture fingerprint; "
                "rename the draft and update its references instead")


def _admit_formal_proposals(bundle, formal_source: FormalSource):
    admitted: list[BundleFile] = []
    notes: list[str] = []
    for proposal in bundle.proposal_files:
        stem = PurePosixPath(proposal.path).stem
        remote = formal_source.components.get(stem)
        digest = hashlib.sha256(remote.body.encode("utf-8")).hexdigest() if remote else None
        if (remote is not None and proposal.formal_body_sha256
                and remote.path == proposal.path
                and digest == proposal.formal_body_sha256):
            admitted.append(proposal)
            notes.append(
                "formal component proposal admitted against exact main baseline: "
                f"{proposal.path}")
        else:
            notes.append(
                "formal proposal omitted from PR: stale/missing/mismatched "
                f"baseline ({proposal.path})")
    return admitted, notes


def spec_pr_branch(model_id: str, record_id: str) -> str:
    """The PR branch of a record: deterministic, so a re-push lands on the
    same PR."""
    return f"spec/{_safe(model_id)}-{record_id[:8]}"


def check_spec_bundle(model_id: str, spec_yaml: str,
                      new_component_yamls: Optional[list]) -> None:
    """Parse and validate the bundle against the deployment: a malformed or
    incomplete bundle fails here, before any network request or job."""
    normalize_bundle(model_id, spec_yaml, new_component_yamls or [],
                     FormalSource.from_deployment())


@dataclass(frozen=True)
class SpecPR:
    """A record's bundle as the PR carries it, computed against one main
    commit: the files, and the title and body that describe them."""

    branch: str
    files: dict
    title: str
    body: str
    commit_prefix: str


def prepare_spec_pr(
    *,
    record_id: str,
    model_id: str,
    spec_yaml: str,
    new_component_yamls: Optional[list] = None,
    snapshot_json: Optional[dict] = None,
    note: str = "",
    contributor: Optional[str] = None,
    formal_source: FormalSource,
) -> SpecPR:
    """The PR's files against ``formal_source`` (main's YAML namespace):
    normalize, reuse the drafts main already holds under their main names,
    check the reuse leaves the architecture unchanged, admit the formal
    proposals main still matches, then lay the files out. The admitter runs
    on exactly these files, so it sees the component names and spec the PR
    carries.

    ``contributor`` credits the user who generated the spec in the body; the
    commit's ``Co-authored-by`` trailer is added when it is published.
    """
    check_spec_bundle(model_id, spec_yaml, new_component_yamls)
    bundle = normalize_bundle(
        model_id,
        spec_yaml,
        new_component_yamls or [],
        formal_source,
    )
    remote_notes = bundle.notes
    control_spec = bundle.spec_yaml
    control_drafts = list(bundle.serving_files)
    bundle, dedupe_map = _dedupe_remote_drafts(bundle, formal_source)
    if dedupe_map:
        control_spec = rename_component_references(
            control_spec, dedupe_map, label="target spec control")
    _verify_remote_reuse(
        model_id,
        (bundle.spec_yaml, list(bundle.serving_files)),
        control_spec,
        control_drafts,
        formal_source,
    )
    proposals, proposal_notes = _admit_formal_proposals(bundle, formal_source)
    notes = [*remote_notes, *bundle.notes, *proposal_notes]
    files = _bundle_files(
        model_id,
        bundle.spec_yaml,
        [file.as_json() for file in [*bundle.serving_files, *proposals]],
        snapshot_json,
        formal_source.registry,
    )
    has_op = _OPS_REGISTRY_REL in files
    title = (
        f"Add spec: {model_id} + primitive op" if has_op
        else f"Add spec: {model_id}"
    )
    pr_body = (
        f"Contributed spec for **{model_id}**, generated via HuggingArch.\n\n"
        + (f"Contributor: {contributor}\n\n" if contributor else "")
        + f"{note}\n\n"
        + (("".join(f"- {item}\n" for item in notes)) + "\n" if notes else "")
        + ("Contains shared-layer component change "
           "(not applied during generation validation).\n\n" if proposals else "")
        +
        f"Files: {', '.join(sorted(files))}\n\n"
        "Merge after CI is green (validate diff / type-contracts / goldens)."
    )
    return SpecPR(branch=spec_pr_branch(model_id, record_id), files=files,
                  title=title, body=pr_body, commit_prefix=f"spec({model_id})")


def input_drafts(new_component_yamls: Optional[list]) -> list[BundleFile]:
    return _input_drafts(new_component_yamls)


_MODEL_STACK_WORKFLOW = "update-transformers.yml"


def trigger_model_stack_update() -> str:
    """Dispatch the model-stack refresh workflow NOW (the admin dashboard
    button). Returns the workflow page URL for the dashboard to link to.

    This only FIRES `.github/workflows/update-transformers.yml` on the default
    branch — the workflow itself re-locks the model-loading stack, runs the
    golden + test gates, and opens a PR. Nothing on this deployment changes
    until a human merges that PR and the deploy pipeline ships it.

    Auth note: dispatching needs **Actions: write** on the repo — for a GitHub
    App, the "Actions" repository permission (the spec-PR flow only needed
    Contents + Pull requests, so an existing App may have to be re-approved);
    for a fine-grained PAT, the "Actions: write" permission.

    Raises GitHubNotConfigured / httpx.HTTPStatusError / httpx.RequestError —
    the caller surfaces them (admin action).
    """
    repo = _repo()
    with httpx.Client(timeout=15.0, headers=_headers()) as cx:
        info = cx.get(f"{_API}/repos/{repo}")
        info.raise_for_status()
        ref = info.json().get("default_branch", "main")
        r = cx.post(
            f"{_API}/repos/{repo}/actions/workflows/{_MODEL_STACK_WORKFLOW}/dispatches",
            json={"ref": ref},
        )
        r.raise_for_status()
    return f"https://github.com/{repo}/actions/workflows/{_MODEL_STACK_WORKFLOW}"


_PR_HTML = re.compile(
    r"https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/pull/(\d+)", re.I)


def pr_merged_state(pr_url: str) -> Optional[str]:
    """Return 'merged' | 'open' | 'closed' for a PR html_url, or None if the
    state can't be read (unconfigured / network / unparseable url).

    Hits the repo named in the URL, not HA_GITHUB_REPO — a contribution
    PR can live on a fork or a renamed repo.
    """
    if not enabled():
        return None
    m = _PR_HTML.search(pr_url or "")
    if not m:
        return None
    owner, repo, num = m.group(1), m.group(2), m.group(3)
    try:
        with httpx.Client(timeout=10.0, headers=_headers()) as cx:
            r = cx.get(f"{_API}/repos/{owner}/{repo}/pulls/{num}")
        if r.status_code != 200:
            logger.warning("pr_merged_state %s -> %s", pr_url, r.status_code)
            return None
        d = r.json()
        if d.get("merged"):
            return "merged"
        return d.get("state") or None
    except httpx.RequestError as e:
        logger.warning(f"pr_merged_state failed: {e}")
        return None
