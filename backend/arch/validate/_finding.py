"""`Finding` — one validator finding, typed instead of a convention-shaped dict.

`kind` is the machine key — ablation levers, tests, and the report's error
formatting all match on it. `detail` is the human text, the only one: a
second text field (`msg`) and a third channel (checks returning bare
strings) used to exist, and the report formatter had to know which one a
finding used. The context fields (`template` / `node` / `op` / `stream` /
`shapes` / `axis`) serialize when set; `extra` carries family-specific keys
(activation's `config_hidden_act`, …) so the report's JSON shape is
unchanged.
`check_crashed` is the kind for a check module that raised: `detail` is
`"{label} raised {ExcType}: {exc}"`, and the Finding lands in that check's
usual list.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Finding:
    kind: str
    detail: str = ""
    template: Optional[str] = None
    node: Optional[str] = None
    op: Optional[str] = None
    stream: Optional[str] = None
    shapes: Optional[List[Any]] = None
    axis: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """The report wire form: every set field, then the extras."""
        out: Dict[str, Any] = {"kind": self.kind}
        if self.detail:
            out["detail"] = self.detail
        for name in ("template", "node", "op", "stream", "shapes", "axis"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out.update(self.extra)
        return out
