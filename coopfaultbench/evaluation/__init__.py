# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Dual-reference (fixed vs union) evaluation of cooperative detection.

The standard toolchain scores a configuration against the UNION of the objects visible to the agents
that are present (the union reference). When a fault removes an agent, that reference shrinks with it.
The FIXED reference keeps the union over the frame's full roster, independent of which agents survive.
Scoring the same detections against both gives AP_union, AP_fixed and dAP = AP_fixed - AP_union.
"""
from .average_precision import voc_ap, pooled_ap
from .reference_sets import union_reference, fixed_reference, greedy_match
from .dual_scoring import DualScore, score_against_references, dual_ap

__all__ = ['voc_ap', 'pooled_ap', 'union_reference', 'fixed_reference', 'greedy_match', 'DualScore',
           'score_against_references', 'dual_ap']
