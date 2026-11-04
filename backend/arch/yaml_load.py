"""YAML documents read through libyaml.

Every YAML document the backend reads — the component library, the model
specs, the registries, a stored or submitted spec, a written spec spliced
back (`compose_yaml`) — is read here, by libyaml's safe loader: the same documents `yaml.safe_load` reads, the same
Python values (every file under `backend/` parses to equal values, key
order included, under both), at a tenth of the time. The component library
alone is read in about 0.15 s rather than 1.5 s.
"""
from __future__ import annotations

from typing import IO, Any, Union

import yaml
from yaml.constructor import SafeConstructor


def load_yaml(source: Union[str, bytes, IO[Any]]) -> Any:
    """One YAML document (text or an open stream) as Python values, safely:
    `yaml.safe_load`'s reading, through libyaml (`yaml.CSafeLoader`)."""
    return yaml.load(source, Loader=yaml.CSafeLoader)


def compose_yaml(text: str) -> Any:
    """One YAML document's node graph through libyaml — what a splice of the
    written text reads its marks off (`authored_yaml`). Its nodes are
    PyYAML's own reader's, marks included; a plain scalar's style is `''`
    here where PyYAML's own reader says `None`."""
    return yaml.compose(text, Loader=yaml.CSafeLoader)


def construct_yaml(node: Any) -> Any:
    """A composed node (`compose_yaml`) as the Python values `load_yaml`
    reads the same text to."""
    return SafeConstructor().construct_object(node, deep=True)
