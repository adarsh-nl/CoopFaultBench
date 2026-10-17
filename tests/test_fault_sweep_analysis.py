# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""The fault-sweep analysis on synthetic rows: completeness fails closed, Holm is step-down, verdicts follow
the interval."""
import json
import os
import subprocess
import sys

import numpy as np

from coopfaultbench.analysis.fault_sweep import (assert_scene_complete, condition_gains, holm_bonferroni,
                                                 scene_verdicts)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cfg(k, tp, gt):
    return {'k': k, 'arm': None, 'realized': [str(i) for i in range(k)], 'score': {'0.7': [0.5] * len(tp)},
            'tp_fixed': {'0.7': tp}, 'fp_fixed': {'0.7': [1 - t for t in tp]}, 'gtn_fixed': {'0.7': gt}}


def arms(n_scenes=6, frames=5, cp_better=True, seed=0):
    g = np.random.default_rng(seed)
    out = {}
    for s in range(n_scenes):
        for f in range(frames):
            gt = 6
            indiv = (g.random(4) < 0.4).astype(int).tolist()
            cp = (g.random(6) < (0.9 if cp_better else 0.4)).astype(int).tolist()
            out[('s%d' % s, f)] = {'individual': cfg(1, indiv, gt), 'cp': cfg(3, cp, gt)}
    return out


def test_holm_is_step_down_and_aligned_to_input_order():
    assert holm_bonferroni([0.01, 0.5, 0.02]) == [True, False, True]
    assert holm_bonferroni([0.04, 0.03]) == [False, False]
    assert holm_bonferroni([0.01, 0.02, 0.03, 0.04], alpha=0.05) == [True, False, False, False]  # 0.02 > 0.05/3
    assert holm_bonferroni([0.01, 0.012, 0.016, 0.04]) == [True, True, True, True]


def test_completeness_fails_closed():
    present = {'a': {1, 2}, 'b': {3}}
    assert assert_scene_complete(present, {'a', 'b'}, {'a': 2, 'b': 1}, 'm', 'c')[0]
    assert not assert_scene_complete(present, {'a', 'b', 'c'}, {'a': 2, 'b': 1, 'c': 1}, 'm', 'c')[0]   # missing
    assert not assert_scene_complete(present, {'a', 'b'}, {'a': 3, 'b': 1}, 'm', 'c')[0]                # short
    assert not assert_scene_complete(present, {'a', 'b'}, {'a': 1, 'b': 1}, 'm', 'c')[0]                # over-count
    assert not assert_scene_complete(present, {'a'}, {'a': 2}, 'm', 'c')[0]                             # unexpected


def test_gain_and_verdicts():
    res = condition_gains({'clean': arms(cp_better=True), 'bad': arms(cp_better=False, seed=1)}, '0.7', 200, 7)
    good = res['clean']
    assert good['n_scenarios'] == 6 and good['gain_ci'][1] > 0 and good['uncorrected_reject']
    assert res == condition_gains({'clean': arms(cp_better=True), 'bad': arms(cp_better=False, seed=1)},
                                  '0.7', 200, 7)                                   # deterministic
    sv = scene_verdicts({'clean': arms(cp_better=True)}, '0.7', 200, 7)['clean']
    assert set(sv) == {'s%d' % i for i in range(6)}
    assert all(v['verdict'] in ('established beneficial', 'not established', 'established harmful', 'N/A')
               for v in sv.values())


def test_scripts_show_help_without_opencood():
    scripts = sorted(f for f in os.listdir(os.path.join(ROOT, 'scripts')) if f.endswith('.py'))
    assert len(scripts) >= 5
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=ROOT)
    for s in scripts:
        cp = subprocess.run([sys.executable, '-B', os.path.join(ROOT, 'scripts', s), '--help'],
                            capture_output=True, text=True, env=env, cwd=ROOT)
        assert cp.returncode == 0, (s, cp.stderr[-400:])
