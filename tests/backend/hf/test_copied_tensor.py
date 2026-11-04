"""A tensor an exporter copied into a second shard is one tensor.

Step-5-Preview keeps `model.embed_tokens.weight` in both its first and its
last shard, and its whole MTP layer in both the last shard and the MTP
shard: 18 names, 2,733,196,800 B of copies. The reader used to count the
tensor set by name and the bytes by occurrence, so the checkpoint's byte
total came out that much too large and the bytes gate failed a spec that
accounted for every tensor exactly once.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.quant_schemes import SCHEME_NONE
from backend.hf.header import ReadTensors
from backend.hf.metadata import scan_local_hf_repo


def _write_shard(path: Path, tensors: dict) -> None:
    """A minimal safetensors file: u64 header length, JSON header, zeros."""
    header = {}
    offset = 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\0" * offset)


EMBED = ("BF16", [8, 4], 8 * 4 * 2)
WEIGHT = ("BF16", [4, 4], 4 * 4 * 2)


class TestACopiedTensorCountsOnce(unittest.TestCase):
    """`ReadTensors` is where "one name is one tensor" is decided, so both
    readers (hub and local repo) count a copy once."""

    def test_the_byte_total_counts_each_name_once(self):
        read = ReadTensors()
        read.add("model.embed_tokens.weight", *EMBED, shard="shard-1")
        read.add("model.layers.0.w.weight", *WEIGHT, shard="shard-1")
        read.add("model.embed_tokens.weight", *EMBED, shard="shard-2")
        self.assertEqual(read.total_bytes, EMBED[2] + WEIGHT[2])
        self.assertEqual(read.names, {"model.embed_tokens.weight",
                                      "model.layers.0.w.weight"})
        self.assertEqual(read.header().total_size, EMBED[2] + WEIGHT[2])

    def test_copies_that_disagree_are_refused(self):
        read = ReadTensors()
        read.add("w", "BF16", [4, 4], 32, shard="shard-1")
        with self.assertRaises(ValueError) as caught:
            read.add("w", "F8_E4M3", [4, 4], 16, shard="shard-2")
        self.assertIn("shard-2", str(caught.exception))
        self.assertIn("disagree", str(caught.exception))

    def test_a_local_repo_with_a_copied_tensor_is_not_double_counted(self):
        """The same checkpoint shape as Step-5-Preview's: the embedding in
        two shards, the index naming one of them."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_shard(repo / "model-00001.safetensors", {
                "model.embed_tokens.weight": EMBED,
                "model.layers.0.w.weight": WEIGHT,
            })
            _write_shard(repo / "model-00002.safetensors", {
                "model.embed_tokens.weight": EMBED,
            })
            (repo / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {
                    "model.embed_tokens.weight": "model-00002.safetensors",
                    "model.layers.0.w.weight": "model-00001.safetensors",
                }}))
            meta = scan_local_hf_repo(repo, scheme=SCHEME_NONE, num_bits=0)
        self.assertEqual(meta["index_total_size"], EMBED[2] + WEIGHT[2])
        self.assertEqual(meta["total_params"], 8 * 4 + 4 * 4)


if __name__ == "__main__":
    unittest.main()
