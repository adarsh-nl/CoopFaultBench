# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Analysis of the five-family fault sweep: cooperative gain per condition and per-scene verdicts.

Everything is scored against the FIXED reference. For each model, IoU and condition (clean + 5 families x
3 tiers = 16):

* condition_gains: the cooperative gain AP_cp - AP_individual (full roster vs ego only) with a scene-level
  bootstrap (B = 1000, seed 20260826): mean, 2.5 / 97.5 percentiles, the same for AP_individual and AP_cp,
  a two-sided bootstrap p-value proxy, and Holm-Bonferroni over the 16 conditions (clean included). A
  condition counts only if its interval excludes 0; a Holm rejection additionally requires that.
  Payload key: coopfaultbench.schema.KEY_PER_CONDITION.
* scene_verdicts: per scene, a frame bootstrap of the gain and a three-way verdict (established beneficial /
  established harmful / not established) from the uncorrected interval, with Holm over the scenes (19 on the V2XSet test split).
  Payload key: coopfaultbench.schema.KEY_PER_SCENE.

AP is arrival-order VOC AP (coopfaultbench.evaluation.voc_ap), which equals OpenCOOD's calculate_ap with
global_sort_detections=False. The functions are those that produced results/payloads/
fault_sweep_fixed_reference.json; their tables print the bootstrap MEANS.
"""

import collections
import glob
import json
import os

import numpy as np

from ..evaluation.average_precision import voc_ap

IOUS = ('0.5', '0.7')
TIERS = ('mild', 'moderate', 'severe')
INJECTORS = ('pose_error', 'latency', 'points_reduce', 'lidar_fog', 'lidar_snow')
CONDITIONS = ['clean'] + ['%s_%s' % (f, t) for f in INJECTORS for t in TIERS]
assert len(CONDITIONS) == 16, len(CONDITIONS)


def _ap(stat):
    return voc_ap(np.asarray(stat['tp'], dtype=np.int64), np.asarray(stat['fp'], dtype=np.int64), stat['gt'])


def blank():
    return {'tp': [], 'fp': [], 'score': [], 'gt': 0}


def extend(stat, iou, cfg, ref='fixed'):
    stat['tp'].extend(cfg['tp_%s' % ref][iou])
    stat['fp'].extend(cfg['fp_%s' % ref][iou])
    stat['score'].extend(cfg['score'][iou])
    stat['gt'] += cfg['gtn_%s' % ref][iou]


def load_jsonl(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        for ln in open(p):
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def expected_scenarios(subset_dir, reference_model='cobevt'):
    """The canonical test scenarios and their frame counts, read from the agent-subset rows of one model
    (one rows_<scenario>.jsonl per scenario). Returns (names, {scenario: n_frames})."""
    names = set()
    frame_counts = {}
    for p in glob.glob(os.path.join(subset_dir, reference_model, 'rows_*.jsonl')):
        base = os.path.basename(p)
        scen = base[len('rows_'):-len('.jsonl')]
        names.add(scen)
        frames = set()
        for ln in open(p):
            if ln.strip():
                frames.add(json.loads(ln)['frame'])
        frame_counts[scen] = len(frames)
    return names, frame_counts


def scenario_frame_sets(arms):
    """{scenario: {frame, ...}} from an arms-like dict keyed by (scenario,
    frame) -- the same key shape clean_arms()/fault_arms() both produce."""
    out = collections.defaultdict(set)
    for scen, fr in arms:
        out[scen].add(fr)
    return out


def assert_scene_complete(scenario_frames_present, expected, expected_frames,
                          model, cond_label):
    """Fail-closed scene completeness. Returns (ok, missing, incomplete).

    A (model, condition) cell is aggregated only if all canonical scenarios are present AND each has
    exactly its expected frame count (an over-count fails too). The caller skips the cell otherwise:
    a bootstrap over a partial scenario roster would look complete while covering a subset.
    scenario_frames_present: {scenario: {frame_id, ...}}; incomplete is {scenario: (have, expected)}."""
    present_names = set(scenario_frames_present)
    missing = expected - present_names
    extra = present_names - expected
    if extra:
        print('  [HALT] %s / %s: %d UNEXPECTED scenario name(s) present, '
              'not among the canonical scenarios: %s -- refusing to aggregate on an '
              'unrecognised roster' % (model, cond_label, len(extra),
                                       sorted(extra)))
        return False, missing, {}
    if missing:
        print('  [HALT] %s / %s: %d canonical scenario(s) MISSING: %s '
              '-- this cell is skipped, not aggregated on a partial roster'
              % (model, cond_label, len(missing), sorted(missing)))
        return False, missing, {}
    incomplete = {}
    for s in sorted(expected):
        want = expected_frames.get(s)
        if want is None:
            continue
        have = len(scenario_frames_present[s])
        if have != want:
            incomplete[s] = (have, want)
    if incomplete:
        detail = ', '.join('%s %d/%d' % (s, h, w)
                           for s, (h, w) in sorted(incomplete.items()))
        print('  [HALT] %s / %s: %d scenario(s) present but frame-count '
              'MISMATCH (have/expected): %s -- this cell is skipped, not '
              'aggregated on a partial-frame scenario' % (model, cond_label,
                                                           len(incomplete), detail))
        return False, set(), incomplete
    return True, set(), {}


def clean_arms(subset_dir, model):
    """{(scenario, frame): {'individual': cfg, 'cp': cfg}} for the clean condition, from the
    agent-subset rows: individual = the ego-only configuration, cp = the largest realised roster."""
    rows = load_jsonl(os.path.join(subset_dir, model, 'rows_*.jsonl'))
    out = {}
    for r in rows:
        cfgs = r.get('configs') or []
        if not cfgs:
            continue
        indiv = next((c for c in cfgs if c['k'] == 1), None)
        cp = max(cfgs, key=lambda c: c['k'])
        if indiv is None:
            continue
        out[(r['scenario'], r['frame'])] = {'individual': indiv, 'cp': cp}
    return out


def fault_arms(fault_dir, model, fault, tier, clean):
    """{(scenario, frame): {'individual': cfg, 'cp': cfg}} for one fault condition. For
    collaborator-only faults the individual arm is identical to clean by construction and is taken
    from `clean`."""
    cond_dir = os.path.join(fault_dir, model, '%s_%s' % (fault, tier))
    rows = load_jsonl(os.path.join(cond_dir, 'rows_*.jsonl'))
    out = {}
    for r in rows:
        cfgs = r.get('configs') or []
        key = (r['scenario'], r['frame'])
        indiv = next((c for c in cfgs if c['arm'] == 'individual'), None)
        cp = next((c for c in cfgs if c['arm'] == 'cp'), None)
        if cp is None:
            continue
        if indiv is None:
            fallback = clean.get(key)
            if fallback is None:
                continue
            indiv = fallback['individual']
        out[key] = {'individual': indiv, 'cp': cp}
    return out


def bins_by_scenario(arms, iou):
    """{scenario: {'individual': stat, 'cp': stat}}, built once."""
    out = collections.defaultdict(lambda: {'individual': blank(), 'cp': blank()})
    for (scen, _fr), a in arms.items():
        extend(out[scen]['individual'], iou, a['individual'])
        extend(out[scen]['cp'], iou, a['cp'])
    return out


def ci(vals, boot):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < boot // 2:
        return None
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_pvalue(vals):
    """Two-sided bootstrap p-value proxy: 2x the smaller tail past zero,
    capped at 1. Used only to feed Holm-Bonferroni's ranking; the reported
    verdict itself is always the CI-excludes-zero rule, never
    the p-value alone."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 1.0
    p_le0 = float(np.mean(v <= 0))
    p_ge0 = float(np.mean(v >= 0))
    return min(1.0, 2 * min(p_le0, p_ge0))


def holm_bonferroni(pvals, alpha=0.05):
    """Standard step-down Holm-Bonferroni. Returns a reject-null list aligned
    to the INPUT order (not the sorted order)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        if pvals[idx] <= thresh:
            reject[idx] = True
        else:
            break   # step-down: once one fails, all later (larger p) fail too
    return reject


def condition_gains(model_arms_by_cond, iou, boot, seed):
    """Cooperative gain AP_cp - AP_individual per condition (scene bootstrap, B replicates; the
    replicate streams are SeedSequence(seed).spawn(B), the same for every condition), plus Holm-
    Bonferroni across the conditions given (the paper: 16, clean included, per model and IoU)."""
    per_cond = {}
    scens_all = sorted({s for arms in model_arms_by_cond.values()
                        for (s, _f) in arms})
    ss = np.random.SeedSequence(seed).spawn(boot)
    for cond, arms in model_arms_by_cond.items():
        binned = bins_by_scenario(arms, iou)
        scens = sorted(binned)
        gains, ap_i, ap_c = [], [], []
        for i in range(boot):
            rng = np.random.default_rng(ss[i])
            if not scens:
                continue
            pick = [scens[j] for j in rng.integers(0, len(scens), len(scens))]
            si, sc = blank(), blank()
            for s in pick:
                b = binned[s]
                for acc, src in ((si, b['individual']), (sc, b['cp'])):
                    acc['tp'].extend(src['tp']); acc['fp'].extend(src['fp'])
                    acc['score'].extend(src['score']); acc['gt'] += src['gt']
            if si['gt'] and sc['gt']:
                api = _ap(si)
                apc = _ap(sc)
                gains.append(apc - api)
                ap_i.append(api)
                ap_c.append(apc)
        c = ci(gains, boot)
        per_cond[cond] = dict(gain_ci=c, n_scenarios=len(scens),
                              p_boot=boot_pvalue(gains),
                              ap_indiv_ci=ci(ap_i, boot),
                              ap_cp_ci=ci(ap_c, boot))
    conds = list(per_cond)
    pvals = [per_cond[c]['p_boot'] for c in conds]
    reject = holm_bonferroni(pvals)
    for c, rj in zip(conds, reject):
        g = per_cond[c]['gain_ci']
        uncorrected = g is not None and not (g[1] <= 0 <= g[2])
        per_cond[c]['uncorrected_reject'] = uncorrected
        per_cond[c]['holm_reject'] = bool(rj) and uncorrected
    return per_cond


def scene_verdicts(model_arms_by_cond, iou, boot, seed):
    """Per-scene three-way verdict, Holm-Bonferroni over the scene family (19 scenes on V2XSet test),
    per (model, condition)."""
    out = {}
    for cond, arms in model_arms_by_cond.items():
        by_scene = collections.defaultdict(list)
        for (scen, fr), a in arms.items():
            by_scene[scen].append((fr, a))
        scenes = sorted(by_scene)
        ss = np.random.SeedSequence(seed).spawn(boot)
        gains_by_scene = {}
        for scen in scenes:
            frames = by_scene[scen]
            gains = []
            for i in range(boot):
                rng = np.random.default_rng(ss[i])
                idxs = rng.integers(0, len(frames), len(frames))
                si, sc = blank(), blank()
                for j in idxs:
                    _fr, a = frames[j]
                    extend(si, iou, a['individual']); extend(sc, iou, a['cp'])
                if si['gt'] and sc['gt']:
                    gains.append(_ap(sc) - _ap(si))
            gains_by_scene[scen] = gains
        pvals = [boot_pvalue(gains_by_scene[s]) for s in scenes]
        reject = holm_bonferroni(pvals)
        per_scene = {}
        for scen, rj in zip(scenes, reject):
            c = ci(gains_by_scene[scen], boot)
            if c is None:
                verdict = 'N/A'
                excl0 = False
            else:
                mean, lo, hi = c
                excl0 = not (lo <= 0 <= hi)
                if excl0 and mean > 0:
                    verdict = 'established beneficial'
                elif excl0 and mean < 0:
                    verdict = 'established harmful'
                else:
                    verdict = 'not established'
            per_scene[scen] = dict(gain_ci=c, verdict=verdict,
                                   uncorrected_reject=excl0,
                                   holm_reject=bool(rj) and excl0)
        out[cond] = per_scene
    return out
