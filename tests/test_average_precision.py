# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
import numpy as np
import pytest

from conftest import needs_opencood_eval
from coopfaultbench.evaluation import pooled_ap, voc_ap


def loop_ap(tp, fp, gt):
    """Plain-Python VOC-2010 AP (reference for the vectorised version)."""
    ctp = cfp = 0
    rec, prec = [], []
    for t, f in zip(tp, fp):
        ctp += int(t)
        cfp += int(f)
        rec.append(ctp / gt)
        prec.append(ctp / (cfp + ctp))
    recalls = [0.0] + rec + [1.0]
    envelope = [0.0] + prec + [0.0]
    running = 0.0
    for j in reversed(range(len(envelope))):          # precision envelope from the right
        running = envelope[j] if envelope[j] > running else running
        envelope[j] = running
    total = 0.0
    for j in range(1, len(recalls)):
        step = recalls[j] - recalls[j - 1]
        if step != 0:
            total += step * envelope[j]
    return total


def test_bit_identical_to_plain_loop():
    g = np.random.default_rng(1)
    for _ in range(2000):
        n = int(g.integers(0, 60))
        tp = (g.random(n) < g.random()).astype(np.int8)
        fp = 1 - tp
        gt = int(tp.sum() + g.integers(1, 10))
        assert voc_ap(tp, fp, gt) == loop_ap(tp, fp, gt)


def test_known_values():
    assert voc_ap([1, 1, 1], [0, 0, 0], 3) == 1.0
    assert voc_ap([1, 1, 1], [0, 0, 0], 4) == 0.75
    assert voc_ap([0, 1], [1, 0], 1) == 0.5
    assert voc_ap([], [], 5) == 0.0
    with pytest.raises(ValueError):
        voc_ap([1], [0], 0)


def test_arrival_order_matters_and_pooling():
    a = ([1, 0], [0, 1], 2)
    b = ([0, 1], [1, 0], 2)
    ab, _ = pooled_ap([a, b])
    ba, _ = pooled_ap([b, a])
    assert ab != ba                       # no global re-sort across frames
    assert pooled_ap([a, b])[1] == 4


@needs_opencood_eval
def test_matches_opencood_calculate_ap():
    from opencood.utils.eval_utils import calculate_ap
    g = np.random.default_rng(2)
    for _ in range(200):
        n = int(g.integers(1, 60))
        tp = (g.random(n) < 0.6).astype(int)
        gt = int(tp.sum() + g.integers(1, 10))
        stat = {0.7: {'tp': list(tp), 'fp': list(1 - tp), 'gt': gt, 'score': list(g.random(n))}}
        assert voc_ap(tp, 1 - tp, gt) == calculate_ap(stat, 0.7, False)[0]
