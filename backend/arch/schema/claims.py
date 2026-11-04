"""A-class golden anchors (Axiom) and the claims sidecar schema."""
from __future__ import annotations

from typing import (List, Literal, Optional)

from pydantic import (BaseModel, ConfigDict, Field)




class Axiom(BaseModel):
    """One A-class golden anchor — a KV fact the modeling SOURCE defines and
    the tool can't self-verify. The inspector declares it (with `src` quoting
    the source line); validate judges it at every call of the component in the
    model's own analysis, against the cache that call grew. The axiom is a
    JUDGE, not a calculator — a mismatch means the SPEC is wrong (fix the
    spec); the axiom never overrides the computation.

    `kind: kv_growth` is the component's KV-cache GROWTH LAW. `storage` is the
    class, checked against each cached tensor's stored-positions expression:
      per_token     — grows with the cache length. Full attention (MHA/GQA/MLA).
      windowed      — caps at a window. SWA. `bound` names the cap.
      constant_state— a recurrent state whose size does not depend on the
                      cache length. Linear/SSM.
      none          — no KV at all.
    `node` names one cached tensor (relative to the component; a nested
    component's tensors carry its name as a prefix), else the claim is over
    all of them. `expect` (optional) is the per-POSITION element count of the
    named tensor, or the total over all, checked for per_token/windowed only.
    """
    model_config = ConfigDict(extra="forbid")

    kind: Literal["kv_growth"]
    node: Optional[str] = None
    expect: Optional[str] = None             # dim expression, resolved in the call's dims
    storage: Optional[Literal["per_token", "windowed", "constant_state", "none"]] = None
    bound: Optional[str] = None              # windowed cap (e.g. "s_window"), resolved in the call's dims
    # Source citation: file::symbol + the VERBATIM source expression (not a
    # line number — line numbers rot when the source shifts; a quoted
    # expression stays grep-able and is what the auditor actually checks).
    src: str = ""



class AxiomSet(BaseModel):
    """A component's A-class golden anchors. KV elements, not bytes
    (dtype-agnostic). Judged in the model's dims at each call; there is no
    reference configuration to state."""
    model_config = ConfigDict(extra="forbid")

    checks: List[Axiom] = []



#: Schema tag on a claims sidecar, so a file that drifts out of this shape
#: fails loudly instead of validating as something else.
CLAIMS_V1 = "huggingarch.claims/v1"



class ComponentClaims(BaseModel):
    """A component's evidence, as its own file.

    `backend/arch/claims/components/<component>.yaml`. Claims live
    beside the component rather than inside it because they are two
    different kinds of statement: the component YAML is the PROGRAM (what
    is computed), the claims file is the EVIDENCE (what the framework
    source says the program must produce). Mixing them meant every read of
    the program scrolled past proofs about it, and a claim could be edited
    as if it were a knob.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_id: Literal[CLAIMS_V1] = Field(alias="schema")
    subject: str
    axioms: AxiomSet
