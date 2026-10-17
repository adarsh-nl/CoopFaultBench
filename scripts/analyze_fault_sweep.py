#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Turn fault-sweep rows into the fault-sweep payload (CPU, no OpenCOOD).

    python scripts/analyze_fault_sweep.py --subset_rows SUBSET_DIR --fault_rows FAULT_DIR --out payload.json

SUBSET_DIR/<model>/rows_*.jsonl   agent-subset rows (scripts/run_agent_subset_sweep.py); they give the clean
                                 condition and the canonical scenarios with their frame counts
FAULT_DIR/<model>/<fault>_<tier>/rows_*.jsonl   fault-sweep rows (scripts/run_fault_sweep.py)

A (model, condition) cell is aggregated only if every canonical scenario is complete; incomplete cells are
listed under 'halted_cells' and the script exits 1. --conditions restricts the analysis to a subset (Holm
flags then cover that subset only; the paper's family is all 16 conditions).
"""
import argparse
import json
import os
import sys

from coopfaultbench.schema import (KEY_PER_CONDITION, KEY_PER_SCENE, RULE_CLEAN_GAIN_NOT_POSITIVE,
                                  RULE_SCENE_HARMFUL)
from coopfaultbench.analysis.fault_sweep import (CONDITIONS, INJECTORS, IOUS, TIERS, assert_scene_complete,
                                                 clean_arms, condition_gains, expected_scenarios, fault_arms,
                                                 scenario_frame_sets, scene_verdicts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--subset_rows', required=True)
    ap.add_argument('--fault_rows', required=True)
    ap.add_argument('--models', default='cobevt,v2xvit,where2comm')
    ap.add_argument('--conditions', default=','.join(CONDITIONS[1:]),
                    help='fault conditions <fault>_<tier> to include (clean is always included)')
    ap.add_argument('--expect_scenarios', type=int, default=19,
                    help='number of canonical scenarios the reference rows must define (V2XSet test: 19; '
                         '0 disables the check)')
    ap.add_argument('--reference_model', default='cobevt',
                    help='model whose agent-subset rows define the canonical scenarios and frame counts')
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20260826)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    wanted = [c for c in a.conditions.split(',') if c]
    expected, expected_frames = expected_scenarios(a.subset_rows, a.reference_model)
    if not expected or (a.expect_scenarios and len(expected) != a.expect_scenarios):
        print('FATAL: %d canonical scenario(s) under %s/%s, expected %d; refusing to analyse a partial roster'
              % (len(expected), a.subset_rows, a.reference_model, a.expect_scenarios), file=sys.stderr)
        return 1
    result = {'boot_M': a.boot, 'seed': a.seed, 'conditions': ['clean'] + wanted, 'models': {}}
    triggers, halted = [], []
    for model in [m.strip() for m in a.models.split(',') if m.strip()]:
        clean = clean_arms(a.subset_rows, model)
        ok, missing, incomplete = assert_scene_complete(scenario_frame_sets(clean), expected, expected_frames,
                                                        model, 'clean')
        if not ok:
            print('FATAL: %s: the clean rows are incomplete (%d missing, %d incomplete scenarios)'
                  % (model, len(missing), len(incomplete)), file=sys.stderr)
            return 1
        arms_by_cond = {'clean': clean}
        for f in INJECTORS:
            for t in TIERS:
                cond = '%s_%s' % (f, t)
                if cond not in wanted:
                    continue
                arms = fault_arms(a.fault_rows, model, f, t, clean)
                ok, missing, incomplete = assert_scene_complete(scenario_frame_sets(arms), expected,
                                                                expected_frames, model, cond)
                if ok:
                    arms_by_cond[cond] = arms
                else:
                    halted.append(dict(model=model, condition=cond, missing_scenarios=sorted(missing),
                                       incomplete_scenarios={s: {'have': h, 'expected': w}
                                                             for s, (h, w) in incomplete.items()}))
        res = {KEY_PER_CONDITION: {}, KEY_PER_SCENE: {}}
        for iou in IOUS:
            per_cond = condition_gains(arms_by_cond, iou, a.boot, a.seed)
            res[KEY_PER_CONDITION][iou] = per_cond
            g = per_cond['clean']['gain_ci']
            if g is not None and g[0] <= 0 and not (g[1] <= 0 <= g[2]):
                triggers.append(dict(rule=RULE_CLEAN_GAIN_NOT_POSITIVE, model=model, iou=iou, mean=g[0],
                                     ci=[g[1], g[2]]))
            per_scene = scene_verdicts(arms_by_cond, iou, a.boot, a.seed)
            res[KEY_PER_SCENE][iou] = per_scene
            for cond, scenes in per_scene.items():
                for scen, d in scenes.items():
                    if d['verdict'] == 'established harmful':
                        triggers.append(dict(rule=RULE_SCENE_HARMFUL, model=model, iou=iou, condition=cond,
                                             scenario=scen, gain_ci=d['gain_ci']))
            print('%s @%s: %d condition(s) analysed' % (model, iou, len(per_cond)), flush=True)
        result['models'][model] = res
    result['triggers'] = triggers
    result['halted_cells'] = halted
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(result, fh, indent=1)
    print('wrote %s; %d trigger(s); %d halted cell(s)' % (a.out, len(triggers), len(halted)))
    return 1 if halted else 0


if __name__ == '__main__':
    sys.exit(main())
