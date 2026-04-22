"""Tests for core/cuckoo_filter.py."""

from __future__ import annotations

import random
import unittest

from vision_mvp.core.cuckoo_filter import CuckooFilter


class TestInsertContainsDelete(unittest.TestCase):
    def test_basic_insert_lookup(self):
        cf = CuckooFilter(capacity=1024, fingerprint_bits=16)
        self.assertTrue(cf.insert("alice"))
        self.assertIn("alice", cf)
        self.assertNotIn("bob", cf)

    def test_no_false_negatives(self):
        cf = CuckooFilter(capacity=1024, fingerprint_bits=16)
        keys = [f"user_{i}" for i in range(500)]
        for k in keys:
            self.assertTrue(cf.insert(k))
        for k in keys:
            self.assertIn(k, cf)

    def test_delete(self):
        cf = CuckooFilter(capacity=1024, fingerprint_bits=16)
        cf.insert("foo")
        self.assertIn("foo", cf)
        self.assertTrue(cf.delete("foo"))
        self.assertNotIn("foo", cf)
        # Second delete fails
        self.assertFalse(cf.delete("foo"))

    def test_len_tracking(self):
        cf = CuckooFilter(capacity=128, fingerprint_bits=16)
        self.assertEqual(len(cf), 0)
        for i in range(10):
            cf.insert(i)
        self.assertEqual(len(cf), 10)
        cf.delete(0)
        self.assertEqual(len(cf), 9)

    def test_load_factor_monotone(self):
        cf = CuckooFilter(capacity=64, fingerprint_bits=16)
        prev = cf.load_factor()
        for i in range(40):
            cf.insert(i)
            cur = cf.load_factor()
            self.assertGreaterEqual(cur, prev - 1e-9)
            prev = cur

    def test_int_keys(self):
        cf = CuckooFilter(capacity=256, fingerprint_bits=16)
        for i in range(100):
            self.assertTrue(cf.insert(i))
        for i in range(100):
            self.assertIn(i, cf)


class TestFalsePositiveRate(unittest.TestCase):
    def test_fpr_bounded(self):
        # At 16-bit fingerprints with 4 slots: theoretical upper bound ~ 1.2e-4
        cf = CuckooFilter(capacity=4096, fingerprint_bits=16)
        inserted = [f"k{i}" for i in range(2000)]
        for k in inserted:
            cf.insert(k)
        # Probe non-members
        probes = [f"probe_{i}" for i in range(50000)]
        false_positives = sum(1 for p in probes if p in cf)
        rate = false_positives / len(probes)
        self.assertLess(rate, 5e-3)  # very slack; actual should be ~1e-4

    def test_expected_fpr_monotone_in_bits(self):
        small = CuckooFilter(capacity=128, fingerprint_bits=8).expected_fpr()
        large = CuckooFilter(capacity=128, fingerprint_bits=16).expected_fpr()
        self.assertGreater(small, large)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_hashes(self):
        a = CuckooFilter(capacity=256, fingerprint_bits=16, seed=42)
        b = CuckooFilter(capacity=256, fingerprint_bits=16, seed=42)
        for i in range(30):
            a.insert(f"x{i}")
            b.insert(f"x{i}")
        # Bucket layouts should match exactly under identical seed.
        self.assertEqual(a._buckets, b._buckets)

    def test_different_seed_different_layout(self):
        a = CuckooFilter(capacity=256, fingerprint_bits=16, seed=1)
        b = CuckooFilter(capacity=256, fingerprint_bits=16, seed=2)
        for i in range(30):
            a.insert(f"x{i}")
            b.insert(f"x{i}")
        self.assertNotEqual(a._buckets, b._buckets)


class TestValidation(unittest.TestCase):
    def test_bad_capacity(self):
        with self.assertRaises(ValueError):
            CuckooFilter(capacity=0)

    def test_bad_fingerprint_bits(self):
        with self.assertRaises(ValueError):
            CuckooFilter(capacity=128, fingerprint_bits=0)
        with self.assertRaises(ValueError):
            CuckooFilter(capacity=128, fingerprint_bits=64)


if __name__ == "__main__":
    unittest.main()
