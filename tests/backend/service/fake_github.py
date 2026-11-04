"""An in-memory GitHub for the spec-PR tests: commits with whole file maps,
branches, pulls, comments and the GraphQL draft mutations, served through
`httpx_mock` as one reusable callback.

`hooks[name]` runs a test's interference at a named point (another push,
main moving, a new review) — the races ADMIT §4.3 asks the Open PR flow to
survive.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import tarfile
from typing import Callable, Dict, List, Optional

from httpx import Request, Response

API = "https://api.github.com"
REPO = "shenh10/HuggingArch"
BASE = f"{API}/repos/{REPO}"


def _sha(*parts: str) -> str:
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()


class FakeGitHub:
    def __init__(self, main_files: Dict[str, str]) -> None:
        self.blobs: Dict[str, str] = {}
        self.trees: Dict[str, Dict[str, str]] = {}
        self.commits: Dict[str, dict] = {}
        self.branches: Dict[str, str] = {}
        self.pulls: List[dict] = []
        self.comments: List[dict] = []
        self.graphql: List[dict] = []
        self.hooks: Dict[str, Callable[["FakeGitHub"], None]] = {}
        self.tarballs: List[str] = []
        self.branches["main"] = self.commit(main_files, parents=[], message="root")

    # ── state helpers a test uses directly ────────────────────────────────
    def commit(self, files: Dict[str, str], *, parents: List[str], message: str) -> str:
        tree = _sha("tree", json.dumps(files, sort_keys=True))
        self.trees[tree] = dict(files)
        sha = _sha("commit", tree, *parents, message, str(len(self.commits)))
        self.commits[sha] = {"tree": tree, "parents": parents, "message": message}
        return sha

    def files(self, sha: str) -> Dict[str, str]:
        return self.trees[self.commits[sha]["tree"]]

    def push(self, branch: str, changes: Dict[str, Optional[str]], message: str = "push") -> str:
        """A commit on `branch` (or main) changing `changes` (None deletes)."""
        head = self.branches[branch]
        files = dict(self.files(head))
        for path, text in changes.items():
            if text is None:
                files.pop(path, None)
            else:
                files[path] = text
        sha = self.commit(files, parents=[head], message=message)
        self.branches[branch] = sha
        return sha

    def pull(self, branch: str) -> Optional[dict]:
        return next((p for p in self.pulls if p["head"] == branch and p["state"] == "open"), None)

    def fire(self, name: str) -> None:
        hook = self.hooks.pop(name, None)
        if hook is not None:
            hook(self)

    def tarball(self, sha: str) -> bytes:
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w:gz") as tar:
            for path, text in sorted(self.files(sha).items()):
                data = text.encode("utf-8")
                info = tarfile.TarInfo(f"shenh10-HuggingArch-{sha[:7]}/{path}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return raw.getvalue()

    # ── the HTTP surface ──────────────────────────────────────────────────
    def install(self, httpx_mock) -> None:
        httpx_mock.add_callback(self.handle, url=re.compile(re.escape(API) + r"/.*"),
                                is_reusable=True)

    def handle(self, request: Request) -> Response:
        path = request.url.path
        method = request.method
        body = json.loads(request.content) if request.content else {}
        params = dict(request.url.params)
        repo = f"/repos/{REPO}"
        if path == "/graphql":
            self.graphql.append(body)
            node = body["variables"]["id"]
            pull = next(p for p in self.pulls if p["node_id"] == node)
            pull["draft"] = "convertPullRequestToDraft" in body["query"]
            return Response(200, json={"data": {}})
        if not path.startswith(repo):
            return Response(404, json={"message": "not found"})
        rest = path[len(repo):]
        if method == "GET" and rest == "":
            return Response(200, json={"default_branch": "main"})
        m = re.fullmatch(r"/git/ref/heads/(.+)", rest)
        if method == "GET" and m:
            sha = self.branches.get(m.group(1))
            if sha is None:
                return Response(404, json={"message": "Not Found"})
            return Response(200, json={"object": {"sha": sha}})
        m = re.fullmatch(r"/tarball/(\w+)", rest)
        if method == "GET" and m:
            self.tarballs.append(m.group(1))
            return Response(200, content=self.tarball(m.group(1)))
        m = re.fullmatch(r"/git/commits/(\w+)", rest)
        if method == "GET" and m:
            return Response(200, json={"tree": {"sha": self.commits[m.group(1)]["tree"]}})
        if method == "POST" and rest == "/git/blobs":
            sha = _sha("blob", body["content"])
            self.blobs[sha] = body["content"]
            return Response(201, json={"sha": sha})
        if method == "POST" and rest == "/git/trees":
            files = dict(self.trees[body["base_tree"]])
            for entry in body["tree"]:
                files[entry["path"]] = self.blobs[entry["sha"]]
            tree = _sha("tree", json.dumps(files, sort_keys=True))
            self.trees[tree] = files
            return Response(201, json={"sha": tree})
        if method == "POST" and rest == "/git/commits":
            sha = _sha("commit", body["tree"], *body["parents"], body["message"],
                       str(len(self.commits)))
            self.commits[sha] = {"tree": body["tree"], "parents": body["parents"],
                                 "message": body["message"]}
            self.fire("after_commit")
            return Response(201, json={"sha": sha})
        if method == "POST" and rest == "/git/refs":
            name = body["ref"].removeprefix("refs/heads/")
            if name in self.branches:
                return Response(422, json={"message": "Reference already exists"})
            self.fire("before_branch_move")
            self.branches[name] = body["sha"]
            self.fire("after_branch_move")
            return Response(201, json={})
        m = re.fullmatch(r"/git/refs/heads/(.+)", rest)
        if method == "PATCH" and m:
            self.fire("before_branch_move")
            self.branches[m.group(1)] = body["sha"]
            self.fire("after_branch_move")
            return Response(200, json={})
        if method == "POST" and rest == "/pulls":
            if self.pull(body["head"]):
                return Response(422, json={"message": "A pull request already exists"})
            number = len(self.pulls) + 1
            pull = {"number": number, "head": body["head"], "state": "open",
                    "title": body["title"], "body": body["body"],
                    "draft": bool(body.get("draft")), "node_id": f"PR_{number}",
                    "html_url": f"https://github.com/{REPO}/pull/{number}"}
            self.pulls.append(pull)
            self.fire("after_pull")
            return Response(201, json=pull)
        if method == "GET" and rest == "/pulls":
            head = params.get("head")
            items = [p for p in self.pulls if p["state"] == "open"
                     and (head is None or head.split(":", 1)[1] == p["head"])]
            if head is None and int(params.get("page", "1")) > 1:
                items = []
            return Response(200, json=[{**p, "head": {"ref": p["head"]}} for p in items])
        m = re.fullmatch(r"/pulls/(\d+)/files", rest)
        if method == "GET" and m:
            return Response(200, json=[])
        m = re.fullmatch(r"/pulls/(\d+)", rest)
        if method == "PATCH" and m:
            pull = self.pulls[int(m.group(1)) - 1]
            pull.update({k: v for k, v in body.items() if k in ("title", "body")})
            return Response(200, json=pull)
        m = re.fullmatch(r"/issues/(\d+)/comments", rest)
        if method == "POST" and m:
            self.comments.append({"number": int(m.group(1)), "body": body["body"]})
            return Response(201, json={})
        m = re.fullmatch(r"/compare/(\w+)\.\.\.(\w+)", rest)
        if method == "GET" and m:
            old, new = self.files(m.group(1)), self.files(m.group(2))
            changed = sorted(p for p in set(old) | set(new) if old.get(p) != new.get(p))
            if int(params.get("page", "1")) > 1:
                changed = []
            return Response(200, json={"files": [{"filename": p} for p in changed]})
        m = re.fullmatch(r"/contents/(.+)", rest)
        if method == "GET" and m:
            text = self.files(self.branches.get(params["ref"], params["ref"])).get(m.group(1))
            if text is None:
                return Response(404, json={"message": "Not Found"})
            return Response(200, json={
                "encoding": "base64",
                "content": base64.b64encode(text.encode("utf-8")).decode()})
        return Response(404, json={"message": f"fake GitHub has no {method} {rest}"})
