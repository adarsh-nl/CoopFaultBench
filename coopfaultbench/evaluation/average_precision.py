# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""VOC-2010 average precision over detections in ARRIVAL order.

The paper's AP values were computed by OpenCOOD's evaluation with ``global_sort_detections=False``:
true/false-positive flags are concatenated frame by frame in the order the frames were scored, and AP
is computed over that sequence WITHOUT re-sorting by score across frames. AP therefore depends on frame
order. This module computes the same quantity from the VOC-2010 definition so that cached match flags can
be re-scored without OpenCOOD:

    recall_i    = cumsum(tp)_i / n_gt
    precision_i = cumsum(tp)_i / (cumsum(tp)_i + cumsum(fp)_i)
    precision is made monotonically non-increasing from the right, and
    AP = sum over the points where recall changes of (recall step) x (precision there),

with the final sum accumulated sequentially in index order. tests/test_average_precision.py checks it
against a plain Python loop bit for bit, and against OpenCOOD's ``calculate_ap`` when OpenCOOD is
importable.
"""
import numpy as np


def voc_ap(tp, fp, n_gt):
    """VOC-2010 AP of one detection sequence (0/1 arrays ``tp``, ``fp``) against ``n_gt`` objects."""
    tp = np.asarray(tp)
    fp = np.asarray(fp)
    if tp.shape != fp.shape:
        raise ValueError('tp and fp must have the same shape')
    if n_gt <= 0:
        raise ValueError('n_gt must be positive; AP is undefined without ground truth')
    ctp = np.cumsum(tp, dtype=np.int64)
    cfp = np.cumsum(fp, dtype=np.int64)
    rec = ctp / n_gt
    with np.errstate(invalid='ignore', divide='ignore'):
        prec = ctp / (cfp + ctp)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
    if idx.size == 0:
        return 0.0
    terms = (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return float(np.add.accumulate(terms)[-1])


def pooled_ap(sequences):
    """AP of several frames' match flags concatenated in the given order.

    ``sequences``: iterable of ``(tp, fp, n_gt)``, one per frame (or configuration). Returns
    ``(ap, total_n_gt)``.
    """
    tps, fps, gt = [], [], 0
    for tp, fp, n in sequences:
        tps.append(np.asarray(tp))
        fps.append(np.asarray(fp))
        gt += int(n)
    tp = np.concatenate(tps) if tps else np.zeros(0, dtype=np.int8)
    fp = np.concatenate(fps) if fps else np.zeros(0, dtype=np.int8)
    return voc_ap(tp, fp, gt), gt
