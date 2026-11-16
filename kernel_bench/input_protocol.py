"""Reproducible RNG streams for rotating and distributed measurements.

The invocation owns the root seed. A standalone launcher may need replicated
weights, rank-local shards and several inputs; this object derives those
streams and records the seeds at the same boundary that seeds the generator.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .invocation import Invocation


@dataclass
class InputProtocol:
    invocation: Invocation
    streams: list[dict[str, Any]] = field(default_factory=list)

    @property
    def input_seed(self) -> int:
        return self.invocation.input_seed

    def manual_seed(self, generator: Any, namespace: str, *,
                    rank: int | None = None, slot: int = 0) -> Any:
        if not namespace or slot < 0 or (rank is not None and rank < 0):
            raise ValueError('RNG streams need a name and nonnegative coordinates')
        payload = '\x1f'.join((str(self.input_seed), namespace,
                                'replicated' if rank is None else str(rank), str(slot)))
        seed = int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), 'big')
        generator.manual_seed(seed)
        self.streams.append(dict(namespace=namespace, rank=rank, slot=slot, seed=str(seed)))
        return generator

    def evidence(self) -> dict[str, Any]:
        if not self.streams:
            raise ValueError('no generator was seeded by this protocol')
        return dict(version=1, input_seed=str(self.input_seed),
                    physical_key=list(self.invocation.physical_key),
                    derivation='uint64_be(blake2b(utf8(root\\x1fnamespace\\x1frank_or_replicated\\x1fslot),digest_size=8))',
                    streams=list(self.streams))

    @staticmethod
    def validate(invocation: Invocation, evidence: dict[str, Any]) -> None:
        if (evidence.get('input_seed') != str(invocation.input_seed)
                or evidence.get('physical_key') != list(invocation.physical_key)
                or not evidence.get('streams')):
            raise ValueError('recorded input protocol does not belong to this invocation')
        protocol = InputProtocol(invocation)
        class SeedOnly:
            def manual_seed(self, value):
                self.value = value
        for entry in evidence['streams']:
            generator = SeedOnly()
            protocol.manual_seed(generator, entry['namespace'],
                                 rank=entry['rank'], slot=entry['slot'])
            if str(generator.value) != entry['seed']:
                raise ValueError('recorded stream seed does not follow the input protocol')
