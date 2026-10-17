# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Dual-reference scoring on a tiny synthetic case with a known answer (release gate G2)."""
import numpy as np
import pytest

from coopfaultbench.evaluation import (DualScore, dual_ap, fixed_reference, greedy_match,
                                       score_against_references, union_reference)

# Two agents. Objects A, B are seen by the ego, C only by the collaborator, D by both.
VISIBLE = {'ego': {'A', 'B', 'D'}, 'cav': {'C', 'D'}}
OBJECTS = ['A', 'B', 'C', 'D']


def iou_matrix(det_hits, reference):
    """IoU 0.9 between a detection and the object it hits, 0 elsewhere (columns = reference objects)."""
    ref = sorted(reference)
    m = np.zeros((len(det_hits), len(ref)))
    for i, obj in enumerate(det_hits):
        if obj in ref:
            m[i, ref.index(obj)] = 0.9
    return m


def test_reference_sets():
    assert union_reference(VISIBLE, ['ego', 'cav']) == set(OBJECTS)
    assert union_reference(VISIBLE, ['ego']) == {'A', 'B', 'D'}
    assert fixed_reference(VISIBLE, ['ego', 'cav']) == set(OBJECTS)


def test_known_answer_collaborator_dropped():
    # The collaborator is dropped; the model, now seeing only the ego's view, detects A, B, D perfectly.
    scores = [0.9, 0.8, 0.7]
    hits = ['A', 'B', 'D']
    union = union_reference(VISIBLE, ['ego'])                 # {A, B, D}: C vanished from the reference
    fixed = fixed_reference(VISIBLE, ['ego', 'cav'])          # {A, B, C, D}
    s = score_against_references(scores, iou_matrix(hits, union), iou_matrix(hits, fixed))
    r = dual_ap([s], iou='0.7')
    assert r['n_gt_union'] == 3 and r['n_gt_fixed'] == 4
    assert r['ap_union'] == 1.0                                # looks perfect against the shrunken reference
    assert r['ap_fixed'] == 0.75                               # C is missed against the fixed reference
    assert r['dap'] == -0.25


def test_no_drop_gives_zero_difference():
    scores = [0.9, 0.8, 0.7, 0.6]
    hits = ['A', 'B', 'C', 'D']
    ref = union_reference(VISIBLE, ['ego', 'cav'])
    s = score_against_references(scores, iou_matrix(hits, ref), iou_matrix(hits, ref))
    r = dual_ap([s])
    assert r['ap_union'] == r['ap_fixed'] == 1.0 and r['dap'] == 0.0


def test_greedy_match_one_detection_per_object():
    tp, fp, order = greedy_match([0.5, 0.9], np.array([[0.8], [0.7]]), 0.5)
    assert list(order) == [1, 0] and list(tp) == [1, 0] and list(fp) == [0, 1]
    with pytest.raises(ValueError):
        greedy_match([0.5], np.zeros((2, 1)), 0.5)


def test_from_row_config():
    cfg = {'tp_union': {'0.7': [1, 1]}, 'fp_union': {'0.7': [0, 0]}, 'gtn_union': {'0.7': 2},
           'tp_fixed': {'0.7': [1, 1]}, 'fp_fixed': {'0.7': [0, 0]}, 'gtn_fixed': {'0.7': 4}}
    r = dual_ap([DualScore.from_row_config(cfg, ious=('0.7',))])
    assert r['ap_union'] == 1.0 and r['ap_fixed'] == 0.5
