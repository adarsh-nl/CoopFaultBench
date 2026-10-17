# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Monte Carlo agent drop over cached agent-subset rows: AP_union, AP_fixed and dAP per drop rate.

Input: one JSONL row per frame (written by ``scripts/run_agent_subset_sweep.py``), holding every
DISTINCT realised agent subset of the frame with its dual-referenced match flags::

    {"frame": int, "scenario": str, "N": int, "ego": str, "roster": [...],
     "configs": [{"realized": [...], "k": int,
                  "tp_union": {"0.5": [...], "0.7": [...]}, "fp_union": {...}, "gtn_union": {...},
                  "tp_fixed": {...}, "fp_fixed": {...}, "gtn_fixed": {...},
                  "score": {"0.5": [...], "0.7": [...]}}, ...]}

Why resampling over cached subsets is exact: the communication-range filter is a per-agent distance
test to the ego, independent of which other agents are present. For a frame whose in-range set is R
(ego included), the realised roster of any kept subset S is (S + ego) intersected with R, and the sweep
enumerated every subset, so every draw has a cached configuration. A missing configuration is a hard
error, never a silent skip.

Draws: for replicate ``rep`` of drop rate tier ``tier``, each non-ego in-range agent is kept iff
``default_rng(blake2s('frame|agent|rep|tier')).random() >= p_drop``. The draws do not depend on the
model, so all models see the same dropped agents. (The paper's payload records ``seed = 20260826``; the
draws do not use it.)

Output per model, tier and IoU (the structure of results/payloads/agent_drop_fixed_vs_union.json):
mean, std and 2.5/97.5 percentiles of dAP over the M replicates, the same for AP_union and AP_fixed, and
per agent-count stratum the mean and std of the stratum dAP. The percentile range reflects only WHICH
agents are dropped, with the frames fixed: it is a Monte Carlo agent-draw range, not a confidence interval
over scenes.
"""
import collections
import glob
import hashlib
import json
import os

import numpy as np

from .average_precision import voc_ap

IOUS = ('0.5', '0.7')
TIERS = collections.OrderedDict([('none', 0.0), ('mild', 0.25), ('moderate', 0.50), ('severe', 0.75)])


def load_rows(model_dir):
    """Rows of ``model_dir/rows_*.jsonl`` (files sorted by name, lines in file order) with configs."""
    rows = []
    for p in sorted(glob.glob(os.path.join(model_dir, 'rows_*.jsonl'))):
        with open(p) as fh:
            for ln in fh:
                if ln.strip():
                    r = json.loads(ln)
                    if r.get('configs'):
                        rows.append(r)
    return rows


def frame_index(row):
    """(R, by_realised): R = the in-range set (realised roster of the first largest config)."""
    by_real, maxk, r_set = {}, -1, None
    for c in row.get('configs', []):
        key = tuple(sorted(c['realized']))
        by_real[key] = c
        if c['k'] > maxk:
            maxk = c['k']
            r_set = set(key)
    return r_set, by_real


def agent_seed(frame_id, agent_id, rep, tier):
    """Seed of one keep/drop draw: blake2s of the explicit key 'frame|agent|rep|tier'."""
    key = '%s|%s|%d|%s' % (frame_id, agent_id, rep, tier)
    return int.from_bytes(hashlib.blake2s(key.encode(), digest_size=8).digest(), 'big')


def draw_kept(frame_id, r_set, ego, p_drop, rep, tier):
    """Realised roster after one i.i.d. agent-drop draw (the ego is never dropped)."""
    kept = {ego}
    for a in sorted(r_set - {ego}):
        if np.random.default_rng(agent_seed(frame_id, a, rep, tier)).random() >= p_drop:
            kept.add(a)
    return tuple(sorted(kept))


def _ap(acc):
    return voc_ap(np.asarray(acc['tp'], dtype=np.int64), np.asarray(acc['fp'], dtype=np.int64), acc['gt'])


def _blank():
    return {'tp': [], 'fp': [], 'gt': 0}


def _pct(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def resample(rows, M=100, tiers=None, ious=IOUS):
    """Monte Carlo agent drop over one model's rows. Returns ``{'n_frames', 'tiers'}``."""
    tiers = TIERS if tiers is None else tiers
    prepped, skipped = [], 0
    for r in rows:
        r_set, by_real = frame_index(r)
        if r_set and r['ego'] in r_set:
            prepped.append((r, r_set, by_real))
        else:
            skipped += 1
    if skipped:
        import warnings
        warnings.warn('%d frame(s) have no usable realised roster (ego not in the largest configuration) and '
                      'are excluded' % skipped, RuntimeWarning, stacklevel=2)
    out = collections.OrderedDict()
    for tier, p in tiers.items():
        iou_res = collections.OrderedDict()
        for iou in ious:
            d_aps, a_us, a_fs, by_n = [], [], [], collections.defaultdict(list)
            for rep in range(M):
                acc_u, acc_f = collections.defaultdict(_blank), collections.defaultdict(_blank)
                pool_u, pool_f = _blank(), _blank()
                for r, r_set, by_real in prepped:
                    key = draw_kept(r['frame'], r_set, r['ego'], p, rep, tier)
                    c = by_real.get(key)
                    if c is None:
                        raise RuntimeError('frame %s: no cached configuration for realised roster %s; the '
                                           'rows do not cover every subset' % (r['frame'], key))
                    n = r['N']
                    for acc, side in ((acc_u[n], 'union'), (acc_f[n], 'fixed'),
                                      (pool_u, 'union'), (pool_f, 'fixed')):
                        acc['tp'].extend(c['tp_%s' % side][iou])
                        acc['fp'].extend(c['fp_%s' % side][iou])
                        acc['gt'] += c['gtn_%s' % side][iou]
                au, af = _ap(pool_u), _ap(pool_f)
                d_aps.append(af - au)
                a_us.append(au)
                a_fs.append(af)
                for n in acc_u:
                    if acc_u[n]['gt'] and acc_f[n]['gt']:
                        by_n[n].append(_ap(acc_f[n]) - _ap(acc_u[n]))
            d_aps, a_us, a_fs = np.array(d_aps), np.array(a_us), np.array(a_fs)
            iou_res[iou] = dict(
                mean=float(d_aps.mean()), std=float(d_aps.std()), ci=_pct(d_aps),
                ap_union=dict(mean=float(a_us.mean()), ci=_pct(a_us)),
                ap_fixed=dict(mean=float(a_fs.mean()), ci=_pct(a_fs)),
                by_stratum={str(n): dict(mean=float(np.mean(v)), std=float(np.std(v)), M=len(v))
                            for n, v in sorted(by_n.items())})
        out[tier] = iou_res
    return {'n_frames': len(prepped), 'tiers': out}
