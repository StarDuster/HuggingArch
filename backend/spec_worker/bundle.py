"""Normalized provenance for generated model-spec bundles.

A stored generation owns one target identity YAML. Agent-authored base files
are run-local scaffolding: at a publish, seed, serving materialization, or PR
boundary they are flattened into that identity (or replaced by a formal base).
This module is that single boundary; callers consume its immutable result
instead of re-selecting paths from a raw database row.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional

import yaml

from backend.arch.authored_yaml import AuthoredYaml
from backend.arch.component_signature import component_structure_signature
from backend.arch.loader import SpecLoadError, merge_base_documents
from backend.arch.spec_migrate import (
    SpecMigrationBlocked,
    migrate_mapping,
    migrate_yaml_text,
    refuse,
    rename_components,
    replaced_keys,
    retired_path_roots,
)

from .outcome import RunErrorKind


class SpecBundleError(ValueError):
    """A bundle's references or file entries are incomplete/ambiguous.

    This is data the owner can repair (rename a draft, recreate a missing
    draft: Agent Fix opens the record for that, see `is_repairable`), or
    regenerate. Schema and semantic spec failures remain loader/validator
    errors.
    """

    def __init__(self, message: str, *, errors: Optional[list[str]] = None,
                 repairable: bool = True):
        super().__init__(message)
        self.errors = list(errors or [message])
        # False only for a stored file list that cannot be written out as it
        # was saved (an entry that is not a file, an unsafe path, one path
        # with two bodies): Agent Fix edits the saved files, so it needs them.
        self.repairable = repairable


#: Where a user takes a repairable refusal. The account page offers Agent
#: Fix on a record whose review ended on NO_PASS or on one of these
#: refusals; Fix stages the record as it was saved and gets the refusals as
#: its work order (`BlockedBundle`).
FIX_PATH = ("on your account page, run Review on the record; it reports "
            "these errors and offers Agent Fix")


def is_repairable(exc: BaseException) -> bool:
    """Whether Agent Fix can open a record `normalize_bundle` refused.

    Repairable: every `SpecBundleError` whose saved files can be staged as
    they were saved (a missing or cyclic base, a missing, unparseable or
    namesake draft, an unparseable spec or `drafts:` list), and a
    `SpecMigrationBlocked` (each refusal names a declaration to write).
    Not repairable: a `SpecBundleError` about the stored file list itself,
    and a plain `SpecLoadError` (a retired syntax the migrator cannot
    rewrite), which has to be regenerated.
    """
    if isinstance(exc, SpecBundleError):
        return exc.repairable
    return isinstance(exc, SpecMigrationBlocked)


def normalize_error_payload(exc: BaseException, *, model_id: str) -> dict:
    """Map a normalize-boundary failure to the Inspector error card.

    Three kinds, because there are three repairs. `SpecBundleError` is a
    bundle reference problem (`spec_bundle_incomplete`). `SpecMigrationBlocked`
    is a rewrite whose every refusal names a declaration to write
    (`spec_needs_declaration`): the record is served again as soon as the
    spec says it. A bare `SpecLoadError` is a schema/load failure with
    nothing to add (`spec_invalid`) — regenerate. `fixable` says whether
    Agent Fix can open the record (`is_repairable`), and only a fixable
    message points there; `errors[]` carries each refusal either way.
    """
    fixable = is_repairable(exc)
    if isinstance(exc, SpecBundleError):
        first = exc.errors[0] if exc.errors else str(exc)
        repair = (f"Agent Fix can repair it ({FIX_PATH}), or regenerate it "
                  "from the Inspector." if fixable
                  else "Regenerate it from the Inspector.")
        return {
            "success": False,
            "error_kind": RunErrorKind.SPEC_BUNDLE_INCOMPLETE.value,
            "status": "bundle_incomplete",
            "message": (
                f"Your saved spec for {model_id} is incomplete: "
                f"{first}. {repair}"
            ),
            "errors": list(exc.errors),
            "fixable": fixable,
        }
    if isinstance(exc, SpecMigrationBlocked):
        from backend.arch.spec_migrate import SCHEMA_ERROR
        return {
            "success": False,
            "error_kind": RunErrorKind.SPEC_NEEDS_DECLARATION.value,
            "status": "spec_needs_declaration",
            "message": SCHEMA_ERROR,
            "errors": list(exc.errors),
            "fixable": fixable,
        }
    if isinstance(exc, SpecLoadError):
        from backend.arch.spec_migrate import REGENERATE_HINT, SCHEMA_ERROR
        detail = str(exc)
        return {
            "success": False,
            "error_kind": RunErrorKind.SPEC_INVALID.value,
            "status": "spec_invalid",
            "message": (detail if detail.startswith(SCHEMA_ERROR)
                        else f"{SCHEMA_ERROR} {REGENERATE_HINT}"),
            "errors": list(getattr(exc, "errors", None) or [detail]),
            "fixable": fixable,
        }
    raise TypeError(
        f"normalize_error_payload expected SpecBundleError or SpecLoadError, "
        f"got {type(exc).__name__}")


@dataclass(frozen=True)
class BundleFile:
    path: str
    body: str
    formal_body_sha256: Optional[str] = None

    def as_json(self) -> dict[str, str]:
        out = {"path": self.path, "body": self.body}
        if self.formal_body_sha256 is not None:
            out["formal_body_sha256"] = self.formal_body_sha256
        return out

    @property
    def draft_name(self) -> Optional[str]:
        """The draft component this file is, under either stored root."""
        return _draft_stem(self.path)

    @property
    def model_name(self) -> Optional[str]:
        """The model spec this file is (an agent base or the identity)."""
        return _model_stem(self.path)


@dataclass(frozen=True)
class FormalComponent:
    tier: str
    body: str
    signature: str
    path: str = ""


@dataclass(frozen=True)
class FormalSource:
    """A read-only view of the formal model/component namespace."""

    models: Mapping[str, str]
    components: Mapping[str, FormalComponent]
    #: The dim-symbol registry (`backend/arch/dims.yaml`) of the same tree;
    #: a spec's own `dim_symbols` are promoted into it. Empty when not read.
    registry: str = ""

    @classmethod
    def from_deployment(
        cls,
        root: Optional[Path] = None,
        *,
        exclude_models: Optional[set[str]] = None,
        exclude_components: Optional[set[tuple[str, str]]] = None,
    ) -> "FormalSource":
        arch_root = Path(root) if root else Path(__file__).resolve().parents[1] / "arch"
        models: dict[str, str] = {}
        excluded_models = exclude_models or set()
        for path in sorted((arch_root / "models").glob("*.yaml")):
            if path.stem in excluded_models:
                continue
            models[path.stem] = path.read_text(encoding="utf-8")
        components: dict[str, FormalComponent] = {}
        excluded_components = exclude_components or set()
        for tier in ("library", "adapters", "drafts"):
            for path in sorted((arch_root / "components" / tier).glob("*.yaml")):
                if (tier, path.stem) in excluded_components:
                    continue
                body = path.read_text(encoding="utf-8")
                components[path.stem] = FormalComponent(
                    tier=tier,
                    body=body,
                    signature=component_structure_signature(body, path=str(path)),
                    path=f"backend/arch/components/{tier}/{path.name}",
                )
        registry_path = arch_root / "dims.yaml"
        registry = (registry_path.read_text(encoding="utf-8")
                    if registry_path.is_file() else "")
        return cls(models=models, components=components, registry=registry)

    def has_component(self, name: str) -> bool:
        return name in self.components


def rename_component_references(text: str, mapping: Mapping[str, str], *,
                                label: str = "bundle YAML") -> str:
    """Rewrite component identities in a stored YAML document's text,
    keeping its comments and written order (`AuthoredYaml`)."""
    if not mapping:
        return text
    return AuthoredYaml(text).rewrite(
        rename_components(_parse_mapping(text, label), mapping))


@dataclass(frozen=True)
class NormalizedBundle:
    model_id: str
    spec_yaml: str
    serving_files: tuple[BundleFile, ...]
    proposal_files: tuple[BundleFile, ...]
    notes: tuple[str, ...]
    formal_source: FormalSource

    def record_files(self, *, include_proposals: bool = True) -> list[dict[str, str]]:
        files = [file.as_json() for file in self.serving_files]
        if include_proposals:
            files.extend(file.as_json() for file in self.proposal_files)
        return files


@dataclass(frozen=True)
class BlockedBundle:
    """A saved record `normalize_bundle` refused with a repairable error
    (`is_repairable`), as Agent Fix opens it.

    The spec and files are exactly as saved: no base flattening, migration
    or draft resolution, because the refusal is about that text and the Fix
    agent edits it. `errors` are the refusals, which are the Fix agent's
    work order. It has the seed surface of a `NormalizedBundle` (identity,
    loader files, formal proposals, `record_files`), and the Fix harvest
    runs the result back through `normalize_bundle`, so a refusal the agent
    left standing is refused again there.
    """

    model_id: str
    spec_yaml: str
    serving_files: tuple[BundleFile, ...]
    proposal_files: tuple[BundleFile, ...]
    errors: tuple[str, ...]

    @classmethod
    def from_refusal(cls, model_id: str, spec_yaml: str, files: Iterable[Any],
                     exc: BaseException) -> "BlockedBundle":
        if not is_repairable(exc):
            raise ValueError(
                f"{type(exc).__name__} is not repairable by Agent Fix: {exc}")
        stored = _validate_paths([_coerce_file(file) for file in files])
        serving = tuple(
            file for path, file in stored.items()
            if not _is_formal_component_path(path))
        proposals = tuple(
            file for path, file in stored.items()
            if _is_formal_component_path(path))
        # Both repairable kinds carry their refusals in `errors`.
        return cls(model_id=model_id, spec_yaml=spec_yaml,
                   serving_files=serving, proposal_files=proposals,
                   errors=tuple(exc.errors))

    def record_files(self, *, include_proposals: bool = True) -> list[dict[str, str]]:
        files = [file.as_json() for file in self.serving_files]
        if include_proposals:
            files.extend(file.as_json() for file in self.proposal_files)
        return files


@dataclass(frozen=True)
class RefusedRecord:
    """A saved record Agent Fix cannot open: `normalize_bundle` refused it
    with an error `is_repairable` rejects. The refusal is carried as a value
    so a caller that only asks "can Fix open this?" reads the answer off the
    type, and one that reports it has the error to report."""

    error: SpecBundleError | SpecLoadError


def review_fingerprint(spec_yaml: str, new_component_yamls: Iterable[Any],
                       config: Optional[Mapping[str, Any]]) -> str:
    """The canonical hash of what a review judged and a PR publishes: the
    record's spec text, its component and op-proposal files (path and body,
    in path order) and the snapshot config validate read. One helper for
    both sides — the review stores it with its verdict, Open PR recomputes
    it from the record — so any difference, however harmless, makes the
    PASS not apply rather than two bundles sharing one fingerprint."""
    files = sorted(
        (str(f.get("path") or ""), str(f.get("body") or ""))
        for f in (new_component_yamls or []) if isinstance(f, Mapping))
    blob = json.dumps(
        {"spec_yaml": spec_yaml or "", "files": files,
         "config": config or {}},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def record_fingerprint(record: Mapping[str, Any]) -> str:
    """`review_fingerprint` of a stored generation record, with the config
    of the snapshot it carries."""
    snapshot = record.get("snapshot_json") or {}
    if isinstance(snapshot, str):
        snapshot = json.loads(snapshot)
    return review_fingerprint(record.get("spec_yaml") or "",
                              record.get("new_component_yamls") or [],
                              snapshot.get("config"))


def open_for_fix(
    model_id: str, spec_yaml: str, files: Iterable[Any],
    formal_source: FormalSource,
) -> "NormalizedBundle | BlockedBundle | RefusedRecord":
    """What Agent Fix makes of a saved record: the normalized bundle when it
    loads, the saved text with its refusals (`BlockedBundle`) when a
    repairable error stops it, and the refusal (`RefusedRecord`) when an
    unrepairable one does."""
    files = list(files or [])
    try:
        return normalize_bundle(model_id, spec_yaml, files, formal_source)
    except (SpecBundleError, SpecLoadError) as exc:
        if is_repairable(exc):
            return BlockedBundle.from_refusal(model_id, spec_yaml, files, exc)
        return RefusedRecord(exc)


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def _coerce_file(value: Any) -> BundleFile:
    if isinstance(value, BundleFile):
        return value
    if not isinstance(value, Mapping):
        raise SpecBundleError("bundle file entries must be mappings",
                             repairable=False)
    path = value.get("path")
    body = value.get("body")
    if not isinstance(path, str) or not path.strip():
        raise SpecBundleError("bundle file entry is missing a non-empty path",
                             repairable=False)
    if not isinstance(body, str):
        raise SpecBundleError(f"bundle file entry {path!r} is missing a string body",
                              repairable=False)
    digest = value.get("formal_body_sha256")
    if digest is not None and not isinstance(digest, str):
        raise SpecBundleError(
            f"bundle file entry {path!r} has a non-string formal_body_sha256",
            repairable=False)
    return BundleFile(path=path, body=body, formal_body_sha256=digest)


def _canonical(doc: Any) -> str:
    """`doc` as one text per meaning: equal exactly when the two documents
    parse to the same values of the same types."""
    return yaml.safe_dump(doc, sort_keys=True, allow_unicode=True)


def _parse_mapping(text: str, label: str) -> dict[str, Any]:
    try:
        value = load_yaml(text) or {}
    except yaml.YAMLError as exc:
        raise SpecBundleError(f"{label} is not parseable YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise SpecBundleError(f"{label} must be a YAML mapping")
    return value


def _canonical_parts(path: str) -> tuple[str, ...]:
    """A stored file path under the roots this repository uses today.

    A record saved before 14aac2b0 (08-28) keeps its files under
    `backend/arch_spec/`, and every check below reads the path to decide what
    a file IS — the target identity, an agent base, a draft, a formal
    proposal. Read through the retired root, or a draft saved then is dropped:
    a hard failure when no library component shares its name, and a silent
    substitution of the library's namesake when one does.
    """
    parts = PurePosixPath(path).parts
    for old, new in retired_path_roots().items():
        if parts[:len(old)] == old:
            return (*new, *parts[len(old):])
    return parts


def _model_stem(path: str) -> Optional[str]:
    parts = _canonical_parts(path)
    if (len(parts) == 4 and parts[:3] == ("backend", "arch", "models")
            and PurePosixPath(path).suffix in (".yaml", ".yml")):
        return PurePosixPath(path).stem
    return None


def _draft_stem(path: str) -> Optional[str]:
    parts = _canonical_parts(path)
    if (len(parts) == 5
            and parts[:4] == ("backend", "arch", "components", "drafts")
            and PurePosixPath(path).suffix in (".yaml", ".yml")):
        return PurePosixPath(path).stem
    return None


def _is_formal_component_path(path: str) -> bool:
    parts = _canonical_parts(path)
    return (len(parts) == 5
            and parts[:3] == ("backend", "arch", "components")
            and parts[3] in ("library", "adapters")
            and PurePosixPath(path).suffix in (".yaml", ".yml"))


def _validate_paths(files: list[BundleFile]) -> dict[str, BundleFile]:
    by_path: dict[str, BundleFile] = {}
    for file in files:
        pure = PurePosixPath(file.path)
        if pure.is_absolute() or ".." in pure.parts:
            raise SpecBundleError(f"unsafe bundle path {file.path!r}",
                                  repairable=False)
        existing = by_path.get(file.path)
        if existing is not None and existing.body != file.body:
            raise SpecBundleError(
                f"bundle contains duplicate path {file.path!r} with different bodies",
                repairable=False)
        by_path[file.path] = file
    return by_path


def normalize_bundle(
    model_id: str,
    spec_yaml: str,
    files: Iterable[Any],
    formal_source: FormalSource,
    *,
    run_models: Optional[Mapping[str, str]] = None,
    run_drafts: Optional[Mapping[str, str]] = None,
) -> NormalizedBundle:
    """Flatten agent bases, rewrite stored schema, resolve draft/proposal files."""
    safe_id = _safe_model_id(model_id)
    raw_files = [_coerce_file(file) for file in files]
    file_map = _validate_paths(raw_files)

    identity_doc = _parse_mapping(spec_yaml, f"target spec for {model_id}")
    base_map: dict[str, str] = {}
    for path, file in file_map.items():
        stem = _model_stem(path)
        if stem is not None and stem != safe_id:
            base_map[stem] = file.body
    base_map.update({
        stem: body for stem, body in (run_models or {}).items()
        if stem != safe_id
    })

    chain: list[str] = []
    flattened: list[str] = []
    while isinstance(identity_doc.get("base"), str) and identity_doc["base"]:
        base_name = identity_doc["base"]
        if base_name in chain or base_name == safe_id:
            path = " -> ".join([*chain, base_name])
            raise SpecBundleError(
                f"cyclic base chain for {model_id}: {path}")
        chain.append(base_name)
        if base_name in formal_source.models:
            if flattened:
                flattened.append(base_name)
            break
        body = base_map.pop(base_name, None)
        if body is None:
            chain_text = " -> ".join(chain)
            raise SpecBundleError(
                f"base component {base_name!r} for {model_id!r} is missing "
                f"(chain {chain_text}); inline the base into the spec or "
                "recreate it, or regenerate the spec")
        parent = _parse_mapping(body, f"agent base {base_name!r} for {model_id!r}")
        flattened.append(base_name)
        identity_doc = merge_base_documents(parent, identity_doc)

    # Draft declarations on the flattened identity are authoritative. A child
    # that omitted `drafts:` inherits its base's opt-in list by shallow merge.
    drafts_value = identity_doc.get("drafts") or []
    if not isinstance(drafts_value, list) or not all(isinstance(v, str) for v in drafts_value):
        raise SpecBundleError(
            f"target spec for {model_id!r} has an invalid top-level drafts list")
    drafts = list(dict.fromkeys(drafts_value))

    run_draft_map = dict(run_drafts or {})
    bundle_draft_map = {
        stem: file.body for path, file in file_map.items()
        if (stem := _draft_stem(path)) is not None
    }
    # What the spec's components and its family's models ARE, for the
    # rewrites that can only be decided by reading them: which dim a called
    # component binds, which switch one of them reads, and how the corpus
    # model of the same family binds that switch. The record's own drafts
    # shadow a library namesake, exactly as the draft resolution below
    # resolves them.
    component_bodies = {
        **{name: component.body
           for name, component in formal_source.components.items()},
        **bundle_draft_map, **run_draft_map,
    }

    identity_doc, identity_migrated = migrate_mapping(
        identity_doc, component_bodies, formal_source.models)
    if identity_migrated.blockers:
        raise refuse(model_id, identity_migrated)

    serving_files: list[BundleFile] = []
    proposal_files: list[BundleFile] = []
    notes: list[str] = list(identity_migrated.notes)

    kept_drafts: list[str] = []
    for name in drafts:
        body = run_draft_map.get(name, bundle_draft_map.get(name))
        formal = formal_source.components.get(name)
        if body is None:
            if formal is None:
                raise SpecBundleError(
                    f"draft component {name!r} for {model_id!r} is not in this "
                    "generation or the current library; recreate the missing "
                    "draft, or regenerate the spec")
            if formal.tier == "drafts":
                kept_drafts.append(name)
            continue
        migrated = migrate_yaml_text(body, component_bodies)
        if not migrated.ok:
            raise refuse(model_id, migrated)
        if migrated.changed:
            body = migrated.text
            notes.extend(migrated.notes)
        try:
            candidate_signature = component_structure_signature(
                body, path=f"draft/{name}.yaml")
        except (ValueError, AttributeError, TypeError) as exc:
            raise SpecBundleError(
                f"draft component {name!r} for {model_id!r} is not parseable: "
                f"{exc}") from exc
        if formal is not None:
            if candidate_signature != formal.signature:
                raise SpecBundleError(
                    f"draft component {name!r} for {model_id!r} conflicts with "
                    "the different formal component of the same name; rename the "
                    "draft and update the spec, or reuse the formal component")
            notes.append(f"reused formal component {name}")
            if formal.tier == "drafts":
                kept_drafts.append(name)
            continue
        kept_drafts.append(name)
        serving_files.append(BundleFile(
            path=f"backend/arch/components/drafts/{name}.yaml", body=body))

    if kept_drafts != drafts:
        identity_doc["drafts"] = kept_drafts

    for path, file in file_map.items():
        stem = _model_stem(path)
        if stem is not None:
            continue  # target duplicates and consumed/unused agent bases never travel
        if _draft_stem(path) is not None:
            continue  # selected drafts were canonicalized above
        if _is_formal_component_path(path):
            proposal_files.append(file)
            notes.append(
                "formal component proposal present but not applied: "
                f"{path}")
            continue
        serving_files.append(file)

    if flattened:
        notes.append("flattened agent base chain: " + " -> ".join(flattened))

    # A spec normalization left as it was is returned as written: its
    # comments are the author's citations, and a reviewer reads them. One it
    # rewrote keeps them too: the rewritten entries go back into the written
    # text, in the order it was written (`AuthoredYaml`).
    if _canonical(load_yaml(spec_yaml)) == _canonical(identity_doc):
        normalized_yaml = spec_yaml
    else:
        normalized_yaml = AuthoredYaml(spec_yaml).rewrite(
            identity_doc, replaced=replaced_keys())
    return NormalizedBundle(
        model_id=model_id,
        spec_yaml=normalized_yaml,
        serving_files=tuple(serving_files),
        proposal_files=tuple(proposal_files),
        notes=tuple(dict.fromkeys(notes)),
        formal_source=formal_source,
    )
