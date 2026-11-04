"""Op-domain workers — the missing-primitive propose → review pipeline.

Three sibling packages, mirroring the `backend.spec_worker` layout but
scoped to extending the primitive-op inventory rather than generating
model specs:

  pipeline/   — background coordinator; watches `logs/op_proposals/` for
                gap files the spec worker drops and drives propose →
                review end-to-end (registered from `backend/api/_app.py`).
  proposal/   — proposer worker; consumes a gap statement and stages a
                candidate op under `backend/arch/ops_proposals/`.
  review/     — reviewer worker; judges a staged proposal, writes
                `review.json`, gates auto-merge via `_rules`.

HTTP wrappers live in `backend/api/routes/op_proposals.py` and
`backend/api/routes/op_reviews.py`; the workers themselves are
transport-agnostic agent runtimes over `backend._agent_runner`.
"""
