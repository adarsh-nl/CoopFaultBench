# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Reference sets for dual-reference scoring, and a minimal greedy matcher.

``visible`` maps each agent id to the set of ground-truth object ids that agent contributes to the
reference (in OpenCOOD: the objects in its annotation within the evaluation range, after its
communication-range filtering).

* union reference  G(A'): objects visible to at least one agent in the evaluated subset A'.
* fixed reference  G(A):  objects visible to at least one agent of the frame's full roster A,
                          the same for every subset of that frame.

So G(A') is a subset of G(A): removing agents can only remove objects from the union reference.
"""
import numpy as np


def union_reference(visible, agents):
    """Union of the object ids visible to ``agents`` (the agent-dependent reference)."""
    out = set()
    for a in agents:
        out |= set(visible.get(a, ()))
    return out


def fixed_reference(visible, full_roster):
    """Union over the frame's full roster (held constant across all subsets of the frame)."""
    return union_reference(visible, full_roster)


def greedy_match(scores, iou, threshold):
    """VOC-style greedy matching of detections to ground truth.

    scores    : (D,) detection scores.
    iou       : (D, G) IoU of every detection with every ground-truth object of the reference.
    threshold : IoU threshold.

    Detections are visited in descending score order; each takes the unmatched ground-truth object with
    the highest IoU if that IoU is >= threshold (true positive), otherwise it is a false positive.
    Returns ``(tp, fp)`` as int8 arrays in descending-score order, plus that order.

    With OpenCOOD installed, coopfaultbench.opencood scores with OpenCOOD's own matcher on rotated boxes;
    this function is for synthetic checks and for users who bring their own IoU matrix.
    """
    scores = np.asarray(scores, dtype=np.float64)
    iou = np.asarray(iou, dtype=np.float64)
    if iou.ndim != 2 or iou.shape[0] != len(scores):
        raise ValueError('iou must be a (D, G) matrix with one row per detection')
    order = np.argsort(-scores, kind='stable')
    taken = np.zeros(iou.shape[1], dtype=bool)
    tp = np.zeros(len(scores), dtype=np.int8)
    fp = np.zeros(len(scores), dtype=np.int8)
    for rank, d in enumerate(order):
        cand = np.where(~taken, iou[d], -1.0)
        g = int(np.argmax(cand)) if cand.size else -1
        if g >= 0 and cand[g] >= threshold:
            taken[g] = True
            tp[rank] = 1
        else:
            fp[rank] = 1
    return tp, fp, order
