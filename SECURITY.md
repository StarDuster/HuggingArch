# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report privately via GitHub's **[Report a vulnerability](https://github.com/shenh10/HuggingArch/security/advisories/new)**
(Security → Advisories on the repository). Include a description, reproduction
steps, and the impact you observed. We'll acknowledge the report and work with
you on a fix and disclosure timeline.

## Scope & notes

HuggingArch analyzes model **configs**, not weights, and the core analyzer runs
no untrusted code. A few areas are security-relevant if you self-host:

- **Agentic spec generation** runs an LLM coding agent as a subprocess. It is
  gated (admin / opt-in), runs under an unprivileged, env-scrubbed process, and
  is **not** meant to be exposed to anonymous users. Do not remove those guards.
- **The public `/analyze` path uses `trust_remote_code=False`.** Keep it that
  way; loading arbitrary HF repo code is an RCE surface.
- **BYOK credentials** are encrypted at rest with `HA_CREDS_ENC_KEY`. Treat that
  key, `SUPABASE_SERVICE_ROLE_KEY`, and the GitHub App private key as secrets;
  never commit them.

Thanks for helping keep the project and its users safe.
