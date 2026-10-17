# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Readers for the files under ``results/payloads``.

One place that knows each payload's layout, so tables and figures agree by
construction on what a cell is and whether it is available.

Fault-sweep payload (``fault_sweep_fixed_reference.json``): a cell that is
absent from the per-condition block ``models.<m>.<KEY_PER_CONDITION>.<iou>``,
or listed in ``halted_cells``, is
reported as ``{'pending': True, 'reason': ...}`` -- callers print that reason,
never a 0.0 or an empty cell. (The released payload has no halted cells.)
"""
from __future__ import annotations

import csv
import json
import os
import re

from . import (AGENT_DROP_PAYLOAD, FAULT_SWEEP_PAYLOAD, GRID_CELLS_DIR,
               GRID_CSV, GRID_MODELS_JSON)

# ------------------------------------------------------------ fault sweep
MODELS = ('cobevt', 'v2xvit', 'where2comm')
INJECTORS = ('pose_error', 'latency', 'points_reduce', 'lidar_fog', 'lidar_snow')
TIERS = ('mild', 'moderate', 'severe')
IOUS = ('0.5', '0.7')
CONDITIONS = ['clean'] + ['%s_%s' % (f, t) for f in INJECTORS for t in TIERS]
assert len(CONDITIONS) == 16, len(CONDITIONS)

COND_FAULT_LABEL = {'pose_error': 'Pose error', 'latency': 'Comm. latency',
                    'points_reduce': 'Points reduce', 'lidar_fog': 'LiDAR fog',
                    'lidar_snow': 'LiDAR snow'}


from ..schema import KEY_PER_CONDITION, KEY_PER_SCENE  # noqa: E402  (names fixed by the payload)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def payload_path(payload_dir, name):
    return os.path.join(payload_dir, name)


def load_fault_sweep(payload_dir):
    return load_json(payload_path(payload_dir, FAULT_SWEEP_PAYLOAD))


def load_agent_drop(payload_dir):
    return load_json(payload_path(payload_dir, AGENT_DROP_PAYLOAD))


def halted_lookup(payload):
    """{(model, condition): halted_cell_dict}."""
    return {(h['model'], h['condition']): h for h in payload['halted_cells']}


def pending_reason(halted_entry):
    inc = halted_entry.get('incomplete_scenarios') or {}
    if inc:
        return 'frame-incomplete: ' + ', '.join(
            '%s %d/%d fr' % (s, d['have'], d['expected'])
            for s, d in sorted(inc.items()))
    missing = halted_entry.get('missing_scenarios') or []
    return '%d/19 scenarios missing' % len(missing)


def row(payload, halted, model, condition, iou):
    """One (model, condition, iou) row -- the payload's per-condition dict
    with 'pending': False added, or {'pending': True, 'reason': ...}."""
    if (model, condition) in halted:
        return {'pending': True,
                'reason': pending_reason(halted[(model, condition)])}
    d = payload['models'].get(model, {}).get(KEY_PER_CONDITION, {}) \
        .get(iou, {}) \
        .get(condition)
    if d is None or d.get('ap_indiv_ci') is None \
            or d.get('ap_cp_ci') is None or d.get('gain_ci') is None:
        return {'pending': True, 'reason': 'no bootstrap data at this cell'}
    out = dict(d)
    out['pending'] = False
    return out


def all_rows(payload):
    """{(model, condition, iou): row_dict} for every cell in the fixed grid."""
    halted = halted_lookup(payload)
    out = {}
    for model in MODELS:
        for cond in CONDITIONS:
            for iou in IOUS:
                out[(model, cond, iou)] = row(payload, halted, model, cond, iou)
    return out


def cond_label(cond):
    if cond == 'clean':
        return 'Clean'
    fault, tier = cond.rsplit('_', 1)
    return '%s %s' % (COND_FAULT_LABEL.get(fault, fault), tier)


# ------------------------------------------------------- inference grid
GRID_FLOAT_COLUMNS = ('clean_ap50', 'clean_ap70', 'faulty_ap50', 'faulty_ap70',
                      'delta_ap70', 'rel_drop_ap50', 'rel_drop_ap70',
                      'n_frames')


def parse_extra(s):
    """``extra_metric`` -> dict. Pairs are separated by ';' or ','."""
    out = {}
    for part in re.split(r'[;,]', s or ''):
        if '=' in part:
            k, v = part.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def load_grid_rows(payload_dir):
    """Grid CSV rows as dicts of STRINGS plus a parsed ``_x`` extra-metric
    dict (what the tables read; numbers are formatted from the strings)."""
    p = payload_path(payload_dir, GRID_CSV)
    with open(p, 'r', encoding='utf-8', newline='') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r['_x'] = parse_extra(r.get('extra_metric', ''))
    return rows


def load_grid_rows_float(payload_dir):
    """Grid CSV rows with the AP/drop/frame columns converted to float
    (NaN when empty) -- what the figures read."""
    p = payload_path(payload_dir, GRID_CSV)
    with open(p, 'r', encoding='utf-8', newline='') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in GRID_FLOAT_COLUMNS:
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                r[k] = float('nan')
        r['_x'] = parse_extra(r.get('extra_metric', ''))
    return rows


def load_grid_models(payload_dir):
    return load_json(payload_path(payload_dir, GRID_MODELS_JSON))


def load_grid_cell(payload_dir, model, injector, tier):
    """One sanitized per-cell result, or None if the cell does not exist."""
    p = os.path.join(payload_dir, GRID_CELLS_DIR, model, injector,
                     tier + '.json')
    if not os.path.exists(p):
        return None
    return load_json(p)
