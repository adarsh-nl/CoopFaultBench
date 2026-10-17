# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Score the same detections against the union and the fixed reference.

The unit of work is a scored CONFIGURATION: one frame evaluated with one realised agent subset. For each
reference and IoU threshold it holds the per-detection match flags (tp, fp, in the detector's output
order) and the reference size n_gt. AP is pooled over configurations in arrival order
(:func:`coopfaultbench.evaluation.average_precision.pooled_ap`).
"""
import numpy as np

from .average_precision import pooled_ap
from .reference_sets import greedy_match

REFERENCES = ('union', 'fixed')


class DualScore(object):
    """Match flags of one configuration against both references, per IoU threshold.

    ``flags[(iou, ref)] = (tp, fp, n_gt)`` with ``iou`` a string such as ``'0.7'``.
    """

    __slots__ = ('flags',)

    def __init__(self, flags):
        self.flags = dict(flags)

    @classmethod
    def from_row_config(cls, cfg, ious=('0.5', '0.7')):
        """Build from one entry of a cached agent-subset row (keys tp_union, fp_union, gtn_union, ...)."""
        flags = {}
        for iou in ious:
            for ref in REFERENCES:
                flags[(iou, ref)] = (cfg['tp_%s' % ref][iou], cfg['fp_%s' % ref][iou],
                                     int(cfg['gtn_%s' % ref][iou]))
        return cls(flags)


def score_against_references(scores, iou_union, iou_fixed, thresholds=(0.5, 0.7)):
    """Match one configuration's detections against both references with :func:`greedy_match`.

    ``iou_union`` / ``iou_fixed``: (D, G) IoU matrices of the detections with the union and the fixed
    reference objects. Returns a :class:`DualScore` (flags in descending-score order).
    """
    flags = {}
    for thr in thresholds:
        key = '%g' % thr                     # '0.5', '0.7' (the row format); 0.25 stays '0.25'
        for ref, m in (('union', iou_union), ('fixed', iou_fixed)):
            tp, fp, _ = greedy_match(scores, m, thr)
            flags[(key, ref)] = (tp, fp, int(np.asarray(m).shape[1]))
    return DualScore(flags)


def dual_ap(scores, iou='0.7'):
    """AP_union, AP_fixed and dAP = AP_fixed - AP_union over a sequence of :class:`DualScore`."""
    scores = list(scores)
    ap_u, gt_u = pooled_ap(s.flags[(iou, 'union')] for s in scores)
    ap_f, gt_f = pooled_ap(s.flags[(iou, 'fixed')] for s in scores)
    return {'ap_union': ap_u, 'ap_fixed': ap_f, 'dap': ap_f - ap_u, 'n_gt_union': gt_u,
            'n_gt_fixed': gt_f}
