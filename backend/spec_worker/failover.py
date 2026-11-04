"""Keep a generate run going when its agent, not its spec, stops it.

Each agent invocation ends with an `agent_end` on its `done`
(`backend.agents.base.AgentEnd`). Three endings do not judge the spec, and
the run continues past them inside its overall deadline, in the same
generate leaf, holding the same lock and log:

  * `agent_unavailable` — the model service refused or failed the request
    (auth, quota, no model configured, 5xx, no connection). The run moves on
    to the next agent backend that has credentials for it, in the order
    `limits.failover_order` gives. Each backend is tried once.
  * `timed_out` — a round's own timer stopped it and the repair rounds are
    spent. When the leaf holds a draft, the same agent continues from it.
  * `context_exhausted` — the agent's session outgrew the model's context
    window. The service answered, so another backend is no remedy; when the
    leaf holds a draft, the same agent continues from it in a new session.

A continuation goes through the resume path (`resume.prepare_resume`):
it re-enters the agent's session when that CLI can and the session can take
another turn, else cold-starts on the task prompt followed by the draft and
its current verdict. With no draft on disk, a failover starts the task
afresh on the next agent.

When nothing is left to try, the run's `done` names why as its
`error_kind` (`RunErrorKind(agent_end)`): `agent_unavailable`, `timed_out`
(the overall deadline is spent; the draft stays resumable), `killed` or
`context_exhausted` (the draft stays resumable, in a new session).

A stopped run — the user's stop, or a lock that was reclaimed — no longer
owns its key and is never continued.

Credentials: a service run stamps `agent_fallbacks` ({cli: env}, env None
for the host's own credentials) — resolved as the run's own were, so a
non-admin never falls back to the host's keys. A run without the stamp is a
host run when it carries no BYOK `agent_env`, and may use any CLI whose host
credentials are set; a BYOK run without the stamp has no fallback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from backend.agents import available_clis, get_backend
from backend.agents.base import AgentEnd, host_has_credentials

from .limits import failover_order
from .overlay import RunEnd, authored_yaml, is_generate_leaf
from .resume import prepare_resume


@dataclass(frozen=True)
class Agent:
    """One agent backend a run may spawn, and the credentials it spawns
    with (None: the host's own credentials for `cli`)."""

    cli: str
    env: Optional[dict]

    @property
    def name(self) -> str:
        return get_backend(self.cli).display_name


class Failover:
    """A generate run's choice of what to do after a `done` that did not
    succeed: continue with an agent (`next`), or end the run there."""

    def __init__(self, model_id: str, *, target_root: Optional[str],
                 task_prep: Optional[Callable[[], dict]] = None):
        """`task_prep` gives the original task's prep for a cold start on a
        draft; None when the run's first prompt is the task itself (a run
        that did not begin by re-entering a session)."""
        self.model_id = model_id
        self.target_root = target_root
        self._task_prep = task_prep
        self._fallbacks: list[Agent] = []
        self.agent: Optional[Agent] = None

    def start(self, prep_result: dict, agent_cli: str) -> None:
        """The run's first agent (`agent_cli` with the run's own
        credentials) and the fallbacks it may move on to."""
        agent = Agent(agent_cli, prep_result.get("agent_env"))
        self.agent = agent
        if self._task_prep is None:
            task_prompt = prep_result.get("prompt") or ""
            self._task_prep = lambda: {"prompt": task_prompt}
        stamped = prep_result.get("agent_fallbacks")
        host_run = prep_result.get("agent_env") is None
        self._fallbacks = []
        for cli in failover_order(agent.cli, available_clis())[1:]:
            if stamped is not None:
                if cli in stamped:
                    self._fallbacks.append(Agent(cli, stamped[cli]))
            elif host_run and host_has_credentials(cli):
                self._fallbacks.append(Agent(cli, None))

    def next(self, done: dict, prep_result: dict) -> Optional[tuple[Agent, str]]:
        """After a `done` that did not succeed: the agent to run next and a
        line saying why, with `prep_result` set up for that continuation —
        or None when the run ends on `done`."""
        if done.get("success") or self.agent is None:
            return None
        if not is_generate_leaf(self.target_root, self.model_id):
            return None
        deadline = prep_result.get("_run_deadline_at")
        if deadline is not None and time.time() >= deadline:
            return None
        end = done.get("agent_end")
        if end == AgentEnd.AGENT_UNAVAILABLE.value:
            if not self._fallbacks:
                return None
            agent = self._fallbacks.pop(0)
            why = (f"{self.agent.name} is unavailable "
                   f"({str(done.get('message') or '')[:200]}); "
                   f"continuing with {agent.name}")
        elif end == AgentEnd.TIMED_OUT.value and authored_yaml(self.target_root):
            agent = self.agent
            why = ("The round reached its time budget; continuing from the "
                   "draft within the run's deadline")
        elif (end == AgentEnd.CONTEXT_EXHAUSTED.value
              and authored_yaml(self.target_root)):
            agent = self.agent
            why = ("The agent's session outgrew its context window; "
                   "continuing from the draft in a new session within the "
                   "run's deadline")
        else:
            return None
        self._continue(agent, done, prep_result)
        self.agent = agent
        return agent, why

    def _continue(self, agent: Agent, done: dict, prep_result: dict) -> None:
        """Point `prep_result` at the continuation `agent` runs: its prompt,
        the session it re-enters (if any) and its credentials. The previous
        attempt's verdict does not carry over."""
        if authored_yaml(self.target_root):
            cont = prepare_resume(
                self.model_id, target_root=self.target_root,
                agent_cli=agent.cli, base_prep=self._task_prep,
                end=RunEnd.from_done(done, prep_result))
        else:
            cont = {**self._task_prep(), "resume_session": None}
        for stale in ("downgrade", "terminal", "_round_timed_out"):
            prep_result.pop(stale, None)
        prep_result.update(cont)
        # The earlier attempt already showed its own notes.
        prep_result["prefetch_logs"] = list(cont.get("prefetch_logs") or [])
        prep_result.update(
            _session_id=cont.get("resume_session"), _agent_cli=agent.cli,
            agent_cli=agent.cli, agent_env=agent.env)
