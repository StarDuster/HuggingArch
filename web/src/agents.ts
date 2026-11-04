/** The coding-agent backends the server registers (`GET /api/agents`, read
 *  from `backend.agents`): which agents a page offers, and the name each is
 *  shown by. Fetched once per page load. */

import { useEffect, useState } from "react";
import { fetchJson } from "./api";

export type AgentBackend = { cli: string; label: string };

export type AgentBackends = {
  agents: AgentBackend[];
  /** The agent a spec run uses when nothing chooses another. */
  default: string;
};

let pending: Promise<AgentBackends> | null = null;

export function loadAgentBackends(): Promise<AgentBackends> {
  if (!pending) {
    pending = fetchJson<AgentBackends>("/api/agents").catch((e: unknown) => {
      pending = null; // a later caller asks again
      throw e;
    });
  }
  return pending;
}

/** The registry, or why it could not be read. */
export function useAgentBackends(): { agents: AgentBackends | null; error: string } {
  const [agents, setAgents] = useState<AgentBackends | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    loadAgentBackends()
      .then((a) => { if (live) setAgents(a); })
      .catch((e: unknown) => { if (live) setError(e instanceof Error ? e.message : String(e)); });
    return () => { live = false; };
  }, []);
  return { agents, error };
}

/** The name an agent is shown by; its id until the registry is read, or for
 *  an id the server no longer registers. */
export function agentLabel(agents: AgentBackends | null, cli?: string | null): string {
  return agents?.agents.find((a) => a.cli === cli)?.label || cli || "";
}
