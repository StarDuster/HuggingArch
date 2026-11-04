"""The text of a YAML document someone wrote, and that text after a rewrite.

A stored spec carries its author's comments: the source citation of a dim,
why a layer is shaped the way it is. A reviewer reads them. The rewrites the
backend applies to a stored document (schema migration, a component rename,
flattening an agent base) work on parsed data, and a parsed document has no
comments. `AuthoredYaml.rewrite(doc)` puts the rewritten document back into
the author's text instead of dumping it fresh:

  * an entry (mapping key or list item) whose value the rewrite left alone
    keeps its written lines, byte for byte, with the comments above it and
    on its lines;
  * an entry the rewrite changed inside keeps its key line and comments, and
    only the changed entries below it are rewritten;
  * an entry whose value was replaced keeps the comments above it and on its
    lines, around the value written in the rewrite format;
  * an entry the rewrite added is written in the rewrite format at the place
    the rewritten document puts it;
  * an entry the rewrite removed takes its value away. Its comments move to
    its replacement only when the caller declares one (`replaced`: old key
    → new key, from the rule that made the replacement), the rewrite added
    that key to the same mapping, and this entry is the key's live
    occurrence and the replacement's only source. Otherwise (a dead
    duplicate of a key written twice, two sources for one replacement, a
    plain drop) they stay at the place it was written: right after the
    entry that preceded it.

So no comment is lost, the written order stays, and only the rewritten
entries change. The text is a function of the written text and the
rewritten document alone. List items are matched by their `name:` when they
have one, else by value and position.

The rewrite format (`dump_rewritten`): written order, block mappings, flow
lists. Text of any shape the splice does not put back (`_Unspliceable`:
anchors and aliases, merge keys, a flow or empty document, a list item it
cannot place) is dumped whole in that format with a warning, and so is any
splice that does not parse back to exactly the rewritten document.
"""
from __future__ import annotations

from backend.arch.yaml_load import compose_yaml, construct_yaml, load_yaml
import bisect
import difflib
import logging
from typing import Any, Mapping, Optional

import yaml

log = logging.getLogger(__name__)

_MERGE_TAG = "tag:yaml.org,2002:merge"


class RewriteDumper(yaml.SafeDumper):
    """Block mappings with flow sequences.

    `stamp_match_model_id` inserts after the `model_type:` key, and keeping
    lists inline makes that anchor stable in rewritten text.
    """

    def represent_list(self, data):
        return super().represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True)


RewriteDumper.add_representer(list, RewriteDumper.represent_list)


class _FlowMapping(dict):
    """A mapping rewritten where the author wrote `{...}`: it stays inline."""


RewriteDumper.add_representer(
    _FlowMapping, lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data, flow_style=True))


def dump_rewritten(doc: Any, *, width: Optional[int] = None) -> str:
    """A document in the rewrite format: written order, flow lists."""
    return yaml.dump(doc, Dumper=RewriteDumper, sort_keys=False,
                     allow_unicode=True, default_flow_style=False, width=width)


def same_document(a: Any, b: Any) -> bool:
    """Equal values of equal types in equal key order (`1 != True`,
    `1 != 1.0`, and a reordered mapping is a different text)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return (len(a) == len(b)
                and all(ka == kb and type(ka) is type(kb) and same_document(va, vb)
                        for (ka, va), (kb, vb) in zip(a.items(), b.items(), strict=True)))
    if isinstance(a, list):
        return len(a) == len(b) and all(map(same_document, a, b))
    return a == b or (a != a and b != b)


class _Unspliceable(Exception):
    """The written text has a shape the splice does not put back."""


class AuthoredYaml:
    """A YAML document as its author wrote it."""

    def __init__(self, text: str):
        self.text = text

    def rewrite(self, doc: Any, *, replaced: Optional[Mapping] = None) -> str:
        """The written text carrying `doc`: see the module docstring.
        `replaced` maps a key a rule removed to the key it wrote in its
        place."""
        try:
            spliced = _Splice(self.text, replaced or {}).document(doc)
        except _Unspliceable as exc:
            log.warning("rewritten YAML dumped without its comments: %s", exc)
            return dump_rewritten(doc)
        try:
            back = load_yaml(spliced)
        except yaml.YAMLError as exc:
            back = exc
        if not same_document(back, doc):
            log.warning("rewritten YAML dumped without its comments: "
                        "the splice did not read back as the rewrite")
            return dump_rewritten(doc)
        return spliced


class _Splice:
    """One pass putting a rewritten document into written lines."""

    def __init__(self, text: str, replaced: Mapping):
        self.replaced_keys = replaced
        self.lines = text.splitlines(keepends=True)
        if self.lines and not self.lines[-1].endswith("\n"):
            self.lines[-1] += "\n"
        try:
            # The text the lines are: ended by a line break.
            self.root = compose_yaml("".join(self.lines))
        except yaml.YAMLError as exc:
            raise _Unspliceable(f"unparseable: {exc}") from exc
        if not isinstance(self.root, yaml.MappingNode) or self.root.flow_style:
            raise _Unspliceable("the document is not a block mapping")
        # Offsets where each line starts, and every scalar's span, so a `#`
        # is a comment only outside a scalar.
        self.line_starts = [0]
        for line in self.lines:
            self.line_starts.append(self.line_starts[-1] + len(line))
        spans: list[tuple[int, int]] = []
        seen: set[int] = set()
        self._index(self.root, spans, seen)
        spans.sort()
        self.span_starts = [s for s, _ in spans]
        self.spans = spans

    # -- reading the written text -------------------------------------------

    def _index(self, node, spans, seen) -> None:
        if id(node) in seen:
            raise _Unspliceable("anchors and aliases")
        seen.add(id(node))
        if isinstance(node, yaml.ScalarNode):
            spans.append((node.start_mark.index, node.end_mark.index))
        elif isinstance(node, yaml.SequenceNode):
            for item in node.value:
                self._index(item, spans, seen)
        elif isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if key.tag == _MERGE_TAG:
                    raise _Unspliceable("merge keys")
                self._index(key, spans, seen)
                self._index(value, spans, seen)

    def value(self, node) -> Any:
        return construct_yaml(node)

    def last_line(self, node) -> int:
        """The last line holding `node`'s text; -1 for an empty value."""
        if isinstance(node, yaml.ScalarNode):
            # A plain scalar (`style` None or '', by reader) left empty.
            if node.value == "" and not node.style:
                return -1
            end = node.end_mark
            return end.line - 1 if end.column == 0 and end.line > node.start_mark.line else end.line
        if node.flow_style:
            return node.end_mark.line
        if isinstance(node, yaml.MappingNode):
            return max((max(self.last_line(k), self.last_line(v), k.start_mark.line)
                        for k, v in node.value), default=node.start_mark.line)
        return max((self.last_line(item) for item in node.value),
                   default=node.start_mark.line)

    def one_line(self, node) -> bool:
        return self.last_line(node) <= node.start_mark.line

    def comment_at(self, line_no: int) -> Optional[str]:
        """The comment on a line (from its `#`), if any."""
        line = self.lines[line_no]
        base = self.line_starts[line_no]
        col = line.find("#")
        while col >= 0:
            if col == 0 or line[col - 1] in " \t":
                pos = base + col
                i = bisect.bisect_right(self.span_starts, pos) - 1
                if not (i >= 0 and self.spans[i][0] <= pos < self.spans[i][1]):
                    return line[col:].rstrip("\r\n")
            col = line.find("#", col + 1)
        return None

    def comments_in(self, first: int, last: int, indent: int) -> list[str]:
        """Every comment on lines `first..last`, as full lines at `indent`."""
        return [" " * indent + comment + "\n"
                for no in range(first, last + 1)
                if (comment := self.comment_at(no)) is not None]

    def orphans(self, start: int, head: int, end: int, indent: int) -> list[str]:
        """The comments of a removed entry: the lines above it as written,
        then the comments on its own lines."""
        return self.lines[start:head] + self.comments_in(head, end, indent)

    def verbatim(self, first: int, last: int) -> list[str]:
        return self.lines[first:last + 1]

    # -- writing ------------------------------------------------------------

    def document(self, doc: Any) -> str:
        if not isinstance(doc, dict) or not doc:
            raise _Unspliceable("the rewrite is not a non-empty mapping")
        written = self.value(self.root)
        if same_document(written, doc):
            return "".join(self.lines)
        body = self.mapping(self.root, written, doc, 0)
        end = self.last_line(self.root)
        return "".join(body + self.lines[end + 1:])

    def mapping(self, node, old: dict, new: dict, start: int) -> list[str]:
        items = node.value
        keys = [self.value(key) for key, _ in items]
        indent = items[0][0].start_mark.column
        entries = []
        first = start
        for key_node, value_node in items:
            head = key_node.start_mark.line
            end = max(head, self.last_line(value_node))
            entries.append((first, head, end, value_node))
            first = end + 1
        # A key written twice reads as its last entry; the earlier one is
        # dead text, and a rewrite inside the mapping drops it like a
        # removed entry, comments kept.
        index = {key: i for i, key in enumerate(keys)}

        def entry(i: int, key: Any, value: Any) -> list[str]:
            first, head, end, value_node = entries[i]
            if same_document(old[key], value):
                return self.verbatim(first, end)
            inner = self.inside(value_node, old[key], value, head + 1)
            if inner is not None:
                return self.verbatim(first, head) + inner
            return self.lines[first:head] + self.replaced(
                head, end, _entry_text(key, _as_written(value_node, value),
                                       self.one_line(value_node)), indent)

        def removed(i: int) -> list[str]:
            first, head, end, _ = entries[i]
            return self.orphans(first, head, end, indent)

        def fresh(key: Any, value: Any) -> list[str]:
            return _indented(dump_rewritten({key: value}), indent)

        order = [(index.get(key), key, value) for key, value in new.items()]
        gone = [i for i, key in enumerate(keys)
                if key not in new or index[key] != i]
        # A removed occurrence goes to its declared replacement only when it
        # is the key's live occurrence and the replacement's one source; a
        # dead duplicate, and every source of a contested target, stays put.
        sources: dict[Any, list[int]] = {}
        for i in gone:
            key = keys[i]
            target = self.replaced_keys.get(key)
            if (index[key] == i and key not in new and target is not None
                    and target in new and target not in index):
                sources.setdefault(target, []).append(i)
        replaced_by = {found[0]: target for target, found in sources.items()
                       if len(found) == 1}
        return self.arrange(order, gone, entry, removed, fresh, replaced_by)

    def sequence(self, node, old: list, new: list, start: int) -> list[str]:
        dash = node.start_mark.column
        entries = []
        first = start
        for item in node.value:
            head = item.start_mark.line
            while not self.lines[head][dash:].startswith("-") or self.lines[head][:dash].strip():
                head -= 1
                if head < first:
                    raise _Unspliceable("a list item without its dash")
            end = max(head, self.last_line(item))
            entries.append((first, head, end, item))
            first = end + 1

        pairs = _match_items(old, new)
        matched = {j: i for i, j in pairs.items()}

        def entry(i: int, j: int, value: Any) -> list[str]:
            first, head, end, item = entries[i]
            if same_document(old[i], value):
                return self.verbatim(first, end)
            inner = self.item_inside(item, head, old[i], value, dash)
            if inner is not None:
                return self.lines[first:head] + inner
            return self.lines[first:head] + self.replaced(
                head, end, _item_text(_as_written(item, value),
                                      self.one_line(item)), dash)

        def removed(i: int) -> list[str]:
            first, head, end, _ = entries[i]
            return self.orphans(first, head, end, dash)

        order = [(matched.get(j), j, value) for j, value in enumerate(new)]
        gone = [i for i in range(len(old)) if i not in pairs]
        return self.arrange(order, gone, entry, removed,
                            lambda _j, value: _indented(_item_text(value), dash))

    def arrange(self, order, gone, entry, removed, fresh,
                replaced_by: Optional[dict] = None) -> list[str]:
        """The entries in the rewrite's order. A removed entry's comments go
        to its declared replacement (`replaced_by`: written index → added
        key); the rest stay right after the kept entry written before them,
        or at the top when none was."""
        replaced_by = replaced_by or {}
        kept = sorted(i for i, _, _ in order if i is not None)
        carried: dict[Any, list[str]] = {}
        after: dict[int, list[str]] = {}
        for i in sorted(gone):
            if i in replaced_by:
                carried.setdefault(replaced_by[i], []).extend(removed(i))
                continue
            anchor = max((k for k in kept if k < i), default=-1)
            after.setdefault(anchor, []).extend(removed(i))
        out: list[str] = list(after.get(-1, []))
        for i, key, value in order:
            if i is None:
                out += carried.get(key, []) + fresh(key, value)
            else:
                out += entry(i, key, value) + after.get(i, [])
        return out

    def inside(self, node, old: Any, new: Any, start: int) -> Optional[list[str]]:
        """A changed value rewritten inside its written block, or None when
        it has to be written whole."""
        if isinstance(node, yaml.ScalarNode) or node.flow_style or not new:
            return None
        if isinstance(node, yaml.MappingNode) and isinstance(old, dict) and isinstance(new, dict):
            return self.mapping(node, old, new, start)
        if isinstance(node, yaml.SequenceNode) and isinstance(old, list) and isinstance(new, list):
            return self.sequence(node, old, new, start)
        return None

    def item_inside(self, item, head: int, old: Any, new: Any, dash: int) -> Optional[list[str]]:
        """A changed list item rewritten inside its written block."""
        if item.start_mark.line != head:
            inner = self.inside(item, old, new, head + 1)
            return None if inner is None else [self.lines[head], *inner]
        if not isinstance(item, yaml.MappingNode):
            return None
        # `- key: ...`: read the dash line as the mapping's first key line,
        # then put the dash back on the first line of the rewritten item.
        written = self.lines[head]
        self.lines[head] = written[:dash] + " " + written[dash + 1:]
        try:
            inner = self.inside(item, old, new, head)
        finally:
            self.lines[head] = written
        if inner is None:
            return None
        for n, line in enumerate(inner):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                if line[dash:dash + 1] != " " or line[:dash].strip():
                    raise _Unspliceable("a list item's first line moved")
                inner[n] = line[:dash] + "-" + line[dash + 1:]
                return inner
        raise _Unspliceable("a list item with no content")

    def replaced(self, head: int, end: int, text: str, indent: int) -> list[str]:
        """An entry written anew in place, keeping the comments on its lines:
        a one-line entry keeps its trailing comment on the line."""
        fresh = _indented(text, indent)
        comments = self.comments_in(head, end, indent)
        if head == end and len(fresh) == 1 and comments:
            return [fresh[0].rstrip("\n") + "  " + comments[0].strip() + "\n"]
        return comments + fresh


def _indented(text: str, indent: int) -> list[str]:
    pad = " " * indent
    return [pad + line if line.strip() else line
            for line in text.splitlines(keepends=True)]


def _as_written(node, value: Any) -> Any:
    """`value` in the style its written node had: a mapping written inline
    is rewritten inline."""
    if (isinstance(node, yaml.MappingNode) and node.flow_style
            and isinstance(value, dict)):
        return _FlowMapping(value)
    return value


def _entry_text(key: Any, value: Any, one_line: bool = False) -> str:
    """A mapping entry in the rewrite format, key at column 0; a value
    written on one line is rewritten on one line."""
    return dump_rewritten({key: value}, width=_ONE_LINE if one_line else None)


def _item_text(value: Any, one_line: bool = False) -> str:
    """A list item in the rewrite format, dash at column 0; an item written
    on one line is rewritten on one line."""
    if isinstance(value, dict) and value and not isinstance(value, _FlowMapping):
        lines = dump_rewritten(value).splitlines(keepends=True)
        return "".join(["- " + lines[0]] + ["  " + line if line.strip() else line
                                            for line in lines[1:]])
    text = dump_rewritten({"x": value}, width=_ONE_LINE if one_line else None)
    assert text.startswith("x:")
    return "-" + text[2:]


#: A line width no rewritten value reaches.
_ONE_LINE = 1 << 30


def _identity(value: Any) -> tuple:
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        return ("name", value["name"])
    return ("value", yaml.safe_dump(value, sort_keys=True, allow_unicode=True))


def _match_items(old: list, new: list) -> dict[int, int]:
    """Written item index -> rewritten item index. Items with a `name:`
    match by name; the rest by value, and an unnamed item the rewrite
    changed matches the one in its place."""
    a = [_identity(v) for v in old]
    b = [_identity(v) for v in new]
    pairs: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.update(zip(range(i1, i2), range(j1, j2), strict=True))
        elif tag == "replace":
            pairs.update(
                (i, j) for i, j in zip(range(i1, i2), range(j1, j2), strict=False)
                if a[i][0] == b[j][0] == "value")
    # A named item the rewrite moved is still that item.
    placed = set(pairs.values())
    by_name = {key: j for j, key in enumerate(b)
               if key[0] == "name" and j not in placed}
    for i, key in enumerate(a):
        if i not in pairs and key in by_name:
            pairs[i] = by_name.pop(key)
    return pairs
