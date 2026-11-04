"""P8: the README's headline counts, derived from the repository.

Each count is computed from the same files the product loads and stated in
both READMEs by a phrase (`PHRASES`). The cells are keyed
`<README>|<count>`; they are not owned by any model. An admission changes a
cell only by the increment its files explain: a model yaml with no `base:`
adds a family, a new `components/library/*.yaml` adds a component, and
nothing an admission brings may move the other four.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

from tools.corpus_pins.corpus import ARCH_DIR, ROOT
from tools.corpus_pins.pin import PinBase, Scope

READMES = ("README.md", "README.zh-CN.md")

# Phrase templates per README, keyed by count name. `{n}` is the number.
PHRASES = {
    "README.md": {
        "families": r"\b{n} architecture families\b",
        "verified_ids": r"\*\*{n} verified model ids\*\*",
        "orgs": r"\bacross {n} orgs\b",
        "components": r"\b{n} reusable component DAGs\b",
        "skus": r"\b{n} SKUs:",
        "fusion_plans": r"\({n} fusion plans\)",
    },
    "README.zh-CN.md": {
        "families": r"{n} 个架构族",
        "verified_ids": r"\*\*{n} 个已验证 model id\*\*",
        "orgs": r"覆盖 {n} 家机构",
        "components": r"{n} 个可复用 comp",
        "skus": r"{n} 种型号：",
        "fusion_plans": r"（{n} 个融合 plan）",
    },
}


def verified_ids() -> List[str]:
    ids = []
    for line in (ARCH_DIR / "supported_models.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def is_family(text: str) -> bool:
    """A model yaml is a family unless it is an identity file (`base:
    <family>`): an identity file pins one model id to an existing family
    and adds no architecture."""
    return not any(line.startswith("base:") for line in text.splitlines())


def family_specs() -> list:
    return [p for p in (ARCH_DIR / "models").glob("*.yaml")
            if is_family(p.read_text(encoding="utf-8"))]


def counts() -> Dict[str, int]:
    ids = verified_ids()
    with (ROOT / "backend" / "inference" / "gpu_specs.csv").open(
            encoding="utf-8") as fh:
        skus = sum(1 for _ in csv.DictReader(fh))
    return {
        "families": len(family_specs()),
        "verified_ids": len(ids),
        "orgs": len({i.split("/", 1)[0] for i in ids}),
        "components": len(list((ARCH_DIR / "components" / "library").glob("*.yaml"))),
        "skus": skus,
        "fusion_plans": len(list(
            (ROOT / "backend" / "inference" / "fusion_plans").glob("*.yaml"))),
    }


def cell(readme: str, name: str) -> str:
    return f"{readme}|{name}"


def _matcher(template: str) -> "re.Pattern[str]":
    return re.compile(template.replace("{n}", r"(\d+)"))


@dataclass(frozen=True)
class ReadmeCountsPin(PinBase):
    readmes: Tuple[str, ...] = READMES

    def artifacts(self) -> Tuple[str, ...]:
        return self.readmes

    def texts(self) -> Dict[str, str]:
        return {name: (ROOT / name).read_text(encoding="utf-8") for name in self.readmes}

    def cells(self, texts: Mapping[str, str]) -> Tuple[Dict[str, int], List[str]]:
        """Each `(README, count)` cell's number, and why a cell cannot be
        read: its phrase must occur exactly once, so a stale count left
        beside a fresh one is not read as the fresh one."""
        values: Dict[str, int] = {}
        problems: List[str] = []
        for readme in self.readmes:
            for name, template in PHRASES[readme].items():
                found = _matcher(template).findall(texts[readme])
                if len(found) != 1:
                    problems.append(
                        f"{readme}: the {name} phrase occurs {len(found)} times")
                    continue
                values[cell(readme, name)] = int(found[0])
        return values, problems

    def verify(self, texts: Mapping[str, str], recomputed: Mapping[str, int]) -> List[str]:
        """Every cell reads exactly once and states the recomputed count."""
        values, problems = self.cells(texts)
        for readme in self.readmes:
            for name in PHRASES[readme]:
                got = values.get(cell(readme, name))
                if got is not None and got != recomputed[name]:
                    problems.append(
                        f"{readme} states {name}={got}; the repository has {recomputed[name]}")
        return problems

    def rewrite(self, texts: Mapping[str, str], deltas: Mapping[str, int]) -> Dict[str, str]:
        """The READMEs with each cell moved by its count's delta. Only the
        digits of a changed cell are replaced."""
        out = dict(texts)
        for readme in self.readmes:
            for name, template in PHRASES[readme].items():
                if not deltas.get(name):
                    continue
                matcher = _matcher(template)
                (match,) = list(matcher.finditer(out[readme]))
                start, end = match.span(1)
                new = int(match.group(1)) + deltas[name]
                out[readme] = out[readme][:start] + str(new) + out[readme][end:]
        return out

    def check_rewrite(self, old: Mapping[str, str], new: Mapping[str, str],
                      deltas: Mapping[str, int]) -> List[str]:
        """`new` is `old` with every cell moved by exactly its delta and no
        other byte changed."""
        before, problems = self.cells(old)
        after, more = self.cells(new)
        problems += more
        if problems:
            return problems
        for key, value in before.items():
            name = key.split("|", 1)[1]
            if after[key] != value + deltas.get(name, 0):
                problems.append(f"P8: {key} {value} -> {after[key]}, delta {deltas.get(name, 0)}")
        if self.rewrite(old, deltas) != dict(new):
            problems.append("P8: README bytes outside the count cells moved")
        return problems


def explained_deltas(before: Mapping[str, int], after: Mapping[str, int],
                     new_model_files: Iterable[str], new_library_files: Iterable[str]) -> Tuple[Dict[str, int], List[str]]:
    """The count deltas an admission brings, and why any is not explained.
    `new_model_files` are the texts of the bundle's new model yamls;
    `new_library_files` the bundle's new `components/library/*.yaml`."""
    deltas = {name: after[name] - before[name] for name in before}
    expected = {name: 0 for name in before}
    expected["families"] = sum(1 for text in new_model_files if is_family(text))
    expected["components"] = len(list(new_library_files))
    problems = [
        f"P8: {name} moves by {deltas[name]}, the bundle explains {expected[name]}"
        for name in sorted(deltas) if deltas[name] != expected[name]]
    return deltas, problems


README_COUNTS = ReadmeCountsPin(
    id="P8",
    title="README counts",
    scope=Scope.REQUIRED_AGGREGATE,
    artifact="README.md",
    owner_tests=(
        "tests/meta/test_readme_counts.py::TestReadmeCounts::test_counts_match_repository",
    ),
)
