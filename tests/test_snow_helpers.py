# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""The clean-room helpers of the snow simulation (``coopfaultbench/injectors/_snow_helpers.py``).

Their outputs were compared with the helpers they replace in the snow code as received via MultiCorrupt (identical on thousands of random inputs,
including the random-number state afterwards; the comparison is in the authors' records). These tests pin the
behaviour that the comparison relied on and that the snow output depends on."""
import hashlib
import json
import random

import numpy as np
import pytest

from coopfaultbench.injectors._snow_helpers import find_sorted_index, ransac_polynomial_fit


def test_find_sorted_index_probe_order_and_edges():
    v = [0.0, 1.0, 2.5, 4.0, 7.0]
    assert [find_sorted_index(v, 0, len(v), t) for t in v] == [0, 1, 2, 3, 4]
    assert find_sorted_index(v, 0, len(v), 3.0) == -1
    assert find_sorted_index([1.0, 1.0, 1.0, 1.0], 0, 3, 1.0) == 1          # first probe (0+3)//2 wins
    assert find_sorted_index(v, 3, 1, 2.5) == -1                           # empty range
    with pytest.raises(IndexError):                                       # callers pass high = len(values)
        find_sorted_index(v, 0, len(v), 100.0)


def test_find_sorted_index_digest():
    """Pure-Python and platform-independent: a digest over 5000 random queries (recorded from the
    helper this function replaces)."""
    r = random.Random(20261017)
    out = []
    for _ in range(5000):
        vals = sorted(set(round(r.uniform(-6.3, 6.3), 3) for _ in range(r.randint(0, 40))))
        tgt = r.choice(vals) if vals and r.random() < 0.6 else round(r.uniform(-7, 7), 3)
        hi = len(vals) if r.random() < 0.5 else r.randint(-1, len(vals))
        try:
            out.append(find_sorted_index(vals, 0, hi, tgt))
        except IndexError:
            out.append('IndexError')
    assert hashlib.sha256(json.dumps(out).encode()).hexdigest() == DIGEST


def test_ransac_random_number_use():
    """Exactly one np.random.randint(len(x), size=n) per iteration, nothing else."""
    rs = np.random.RandomState(3)
    x = np.sort(rs.uniform(10, 70, 120))
    y = 0.002 * x ** 2 - 0.1 * x + 6 + rs.normal(0, 0.02, 120)
    y[::9] += 4.0
    np.random.seed(11)
    fit = ransac_polynomial_fit(x, y, order=2)
    after = np.random.get_state()
    np.random.seed(11)
    for _ in range(100):
        np.random.randint(120, size=15)
    ref = np.random.get_state()
    assert after[0] == ref[0] and np.array_equal(after[1], ref[1]) and after[2:] == ref[2:]
    assert np.allclose(fit, [0.002, -0.1, 6.0], atol=0.02)                # the outliers are rejected
    assert not np.allclose(np.polyfit(x, y, 2), fit, atol=1e-3)           # ... which plain least squares does not


DIGEST = '0fccb809fd60020122d4bbc48ab0bdb7b3a41c34605ef9417dee5f627617511a'
