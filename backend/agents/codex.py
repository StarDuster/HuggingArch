"""Codex CLI backend."""
from __future__ import annotations

import json
import re
import subprocess
from typing import List, Optional

from .base import AgentBackend
from .usage import extract_usage_events


class CodexBackend(AgentBackend):
    cli_name = "codex"
    display_name = "Codex"
    supports_resume = True
    # `codex exec --json` ends a refused request on `turn.failed` quoting the
    # HTTP status (`unexpected status 401 Unauthorized: Missing bearer or
    # basic authentication in header, url: https://api.openai.com/v1/responses`,
    # `exceeded retry limit, last status: 429 Too Many Requests`), and its
    # transport logs refusals on stderr (`failed to connect to websocket:
    # HTTP error: 401 Unauthorized`). A spent plan or a dropped connection is
    # named in words.
    unavailable_errors = (
        re.compile(r"(?:unexpected status|last status:|HTTP error:) (?P<status>\d{3})\b"),
        re.compile(r"usage limit|quota exceeded|insufficient_quota"
                   r"|stream disconnected before completion|error sending request",
                   re.I),
    )
    # A thread that outgrew the context window ends on `turn.failed` with
    # codex's own words for it (codex-cli 0.149: "Codex ran out of room in
    # the model's context window. Start a new thread or clear earlier
    # history before retrying.").
    context_exhausted_errors = (
        re.compile(r"ran out of room in the model's context window"),
    )

    @classmethod
    def _merge_env(cls, env: Optional[dict]) -> dict:
        """Add the current Codex CLI's API-key alias.

        Credential storage deliberately speaks the provider-neutral names in
        AGENT_ENV_VARS. Codex 0.149 still discovers OPENAI_API_KEY for some
        checks, but its Responses/WebSocket transport authenticates with
        CODEX_API_KEY. Without this alias the transport reports “auth mode:
        none” and OpenAI rejects the request for a missing bearer token.
        """
        child_env = super()._merge_env(env)
        if child_env.get("OPENAI_API_KEY"):
            child_env["CODEX_API_KEY"] = child_env["OPENAI_API_KEY"]
        return child_env

    @staticmethod
    def _user_endpoint_args(env: dict) -> list[str]:
        """Express a BYOK endpoint as Codex config overrides.

        Codex does not apply OPENAI_BASE_URL to its active model provider when
        it runs with the scrubbed per-agent HOME used here. A provider override
        keeps the endpoint explicit for both spawn and resume without exposing
        the host's ~/.codex/config.toml or auth.json to untrusted model code.
        """
        args: list[str] = []
        base_url = (env.get("OPENAI_BASE_URL") or "").strip()
        if base_url:
            provider = "huggingarch_user"
            args.extend([
                "-c", f'model_provider="{provider}"',
                "-c", f'model_providers.{provider}.name="HuggingArch user endpoint"',
                "-c", f"model_providers.{provider}.base_url={json.dumps(base_url)}",
                "-c", f'model_providers.{provider}.wire_api="responses"',
                "-c", f"model_providers.{provider}.requires_openai_auth=true",
            ])
        model = (env.get("OPENAI_MODEL") or "").strip()
        if model:
            args.extend(["-m", model])
        return args

    def _exec(self, argv: list, prompt: str, env: Optional[dict],
              run_cwd: str, child_env: dict) -> subprocess.Popen:
        proc = subprocess.Popen(
            argv,
            cwd=run_cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=child_env,
            **self._priv_kwargs(env),
        )
        if proc.stdin is not None:
            proc.stdin.write(prompt)
            proc.stdin.close()
        return proc

    def spawn(self, prompt: str, env: Optional[dict] = None,
              workspace: Optional[str] = None) -> subprocess.Popen:
        # `--dangerously-bypass-approvals-and-sandbox` instead of `--full-auto`:
        # codex's own sandbox uses Bubblewrap, which fails to launch inside our
        # already-sandboxed environment (`bwrap: Failed to make / slave:
        # Permission denied`) — so every command (file reads, `validate`) errors
        # and the agent exits having done nothing. codex's own help notes this
        # flag is "intended solely for running in environments that are
        # externally sandboxed", which is exactly us: spec-gen runs codex as the
        # unprivileged HA_AGENT_USER with a scrubbed (secret-free) env, so codex's
        # redundant inner sandbox is safe to drop.
        #
        # Codex has no path-scoped tool policy. Overlay runs still start: cwd
        # and `-C` are the kind leaf, HA_SPEC_OVERLAY_DIR points there, and
        # the prompt pins the write surface. Isolation is not claimed.
        run_cwd, child_env = self._run_cwd_and_env(env, workspace)
        return self._exec(
            ["codex", "exec", "--json",
             *self._user_endpoint_args(child_env),
             "--dangerously-bypass-approvals-and-sandbox",
             "-C", run_cwd, "-"],
            prompt, env, run_cwd, child_env)

    def spawn_resume(self, session_id: str, prompt: str,
                     env: Optional[dict] = None,
                     workspace: Optional[str] = None) -> subprocess.Popen:
        """`codex exec resume <thread_id> -` — the follow-up turn lands in the
        same thread, so a repair round keeps full context.

        Workspace is Popen cwd plus HOME/TMPDIR/HA_SPEC_OVERLAY_DIR.
        `exec resume` does not accept `-C` (codex-cli 0.139+/0.149: unexpected
        argument, exit 2); the initial `codex exec` spawn still passes `-C`.
        """
        run_cwd, child_env = self._run_cwd_and_env(env, workspace)
        return self._exec(
            ["codex", "exec", "resume", "--json",
             *self._user_endpoint_args(child_env),
             "--dangerously-bypass-approvals-and-sandbox",
             session_id, "-"],
            prompt, env, run_cwd, child_env)

    def extract_session_id(self, evt: dict) -> Optional[str]:
        if evt.get("type") == "thread.started":
            tid = evt.get("thread_id")
            return str(tid) if tid else None
        return None

    def parse_event(self, evt: dict) -> List[dict]:
        results = extract_usage_events(evt)
        etype = evt.get("type", "")

        if etype == "thread.started":
            results.append({"type": "log", "text": "Coding agent connected"})
            return results

        if etype == "turn.started":
            return results

        if etype == "error":
            text = evt.get("message", "").strip()
            if text:
                results.append({"type": "log", "text": text})
            return results

        if etype == "turn.failed":
            err = evt.get("error", {})
            if isinstance(err, dict):
                message = err.get("message", "Failed")
            else:
                message = str(err or "Failed")
            results.append({"type": "done", "success": False, "message": message[:300]})
            return results

        if etype in {"turn.completed", "turn.complete", "turn.succeeded"}:
            results.append({"type": "done", "success": True})
            return results

        text = _extract_codex_event_text(evt)
        if text:
            event = {"type": "log", "text": text}
            if _is_agent_message(evt):
                event["author"] = "agent"
            results.append(event)

        return results


def _is_agent_message(evt: dict) -> bool:
    """Whether the event is the model's own reply: `codex exec --json` emits
    it as `item.completed` with `item.type == "agent_message"` (reasoning
    and command items are other types)."""
    item = evt.get("item")
    return isinstance(item, dict) and item.get("type") == "agent_message"


def _extract_codex_event_text(evt: dict) -> str:
    etype = evt.get("type", "")
    if etype.endswith(".delta"):
        return ""

    direct = evt.get("message")
    if isinstance(direct, str):
        return direct.strip()

    for key in ("item", "output_item", "response"):
        value = evt.get(key)
        text = _extract_text_from_obj(value)
        if text:
            return text

    return ""


def _extract_text_from_obj(obj) -> str:
    if isinstance(obj, str):
        return obj.strip()

    if not isinstance(obj, dict):
        return ""

    cmd = obj.get("command")
    if isinstance(cmd, str) and cmd.strip():
        return f"$ {cmd.strip()[:120]}"

    name = obj.get("name")
    item_type = obj.get("type", "")
    if isinstance(name, str) and item_type in {"tool_call", "function_call"}:
        return f"[tool] {name}"

    text = obj.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    message = obj.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()

    content = obj.get("content")
    if isinstance(content, list):
        parts = []
        for entry in content:
            if not isinstance(entry, dict):
                continue
            entry_text = entry.get("text")
            if isinstance(entry_text, str) and entry_text.strip():
                parts.append(entry_text.strip())
        if parts:
            return "\n".join(parts)

    return ""
