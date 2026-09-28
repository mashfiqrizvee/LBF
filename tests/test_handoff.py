from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbf_handoff.bloom import (
    BloomBank, OriginalBloomBank, bloom_parameters, encode_blocks, original_address,
)
from lbf_handoff.data import read_samples, validate_protocol
from lbf_handoff.pipeline1 import layout as p1_layout
from lbf_handoff.pipeline2 import layout as local_layout
from lbf_handoff.pipeline2 import zigzag_indices


class HandoffTests(unittest.TestCase):
    def test_manifest_and_identity_boundaries(self) -> None:
        rows = read_samples(ROOT / "data" / "selected_samples.csv")
        summary = validate_protocol(rows)
        self.assertEqual(sum(item["images"] for item in summary.values()), 9742)
        self.assertEqual(sum(item["identities"] for item in summary.values()), 1498)

    def test_p1_original_and_corrected_layouts(self) -> None:
        original, thesis = p1_layout("original"), p1_layout("thesis")
        self.assertEqual(len(original), 165)
        self.assertEqual(len(thesis), 167)
        self.assertEqual(sum(block.level == "lowest" for block in thesis), 154)
        self.assertEqual(sum(block.level == "middle" for block in thesis), 12)
        self.assertEqual(thesis[-1].length, 4900)

    def test_local_layout_is_shared(self) -> None:
        landmarks = json.loads((ROOT / "frozen" / "selected_landmarks.json").read_text())["selected_landmarks"]
        blocks = local_layout(landmarks)
        self.assertEqual(len(blocks), 48)
        self.assertTrue(all(block.length == 12 for block in blocks))
        self.assertEqual(zigzag_indices()[:5], [(0, 1), (1, 0), (0, 2), (1, 1), (2, 0)])

    def test_small_bloom_round_trip(self) -> None:
        blocks = local_layout([0, 1], bits=3)
        bits = np.asarray([[0, 0, 0, 1, 0, 1], [1, 1, 1, 0, 1, 0]], dtype=bool)
        codes = encode_blocks(bits, blocks)
        bank = BloomBank(blocks, population=1, probability=0.01)
        bank.enroll(codes, [0])
        self.assertEqual(bank.query(codes, [0]).tolist(), [2])
        self.assertLessEqual(bank.query(codes, [1])[0], 2)
        self.assertEqual((bank.m, bank.k), bloom_parameters(1, 0.01))

    def test_original_hash_fixture(self) -> None:
        # Fixture calculated by the supplied 2019 SHA256 -> fnv.hash -> XOR
        # folding code.  It protects against making the old routine "nicer."
        self.assertEqual(original_address(np.asarray([1, 0, 1]), 1, 140_530), 2601)

    def test_original_bank_round_trip(self) -> None:
        bits = np.zeros((2, 4900), dtype=bool)
        bits[1, :32] = True
        bank = OriginalBloomBank(p1_layout("original"), filter_size=16, hash_count=2)
        bank.enroll(bits, [0])
        self.assertEqual(bank.query(bits, [0]).tolist(), [165])
        self.assertLessEqual(bank.query(bits, [1])[0], 165)


if __name__ == "__main__":
    unittest.main()
