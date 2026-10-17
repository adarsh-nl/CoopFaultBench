# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Monte Carlo agent drop over cached agent-subset rows (synthetic rows)."""
import itertools

import numpy as np
import pytest

from coopfaultbench.evaluation.agent_drop import TIERS, draw_kept, resample


def synthetic_rows(n_frames=12, seed=0):
    """Frames with 2-3 agents; every subset containing the ego has a cached configuration."""
    g = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        agents = ['e', 'a', 'b'][:2 + f % 2]
        configs = []
        for k in range(len(agents)):
            for combo in itertools.combinations(agents[1:], k):
                realized = sorted(['e'] + list(combo))
                n = int(g.integers(1, 6))
                tp = (g.random(n) < 0.7).astype(int).tolist()
                fp = [1 - t for t in tp]
                gt_fixed = 6
                gt_union = 3 + len(combo)
                c = {'realized': realized, 'k': len(realized), 'score': {}}
                for ref, gt in (('union', gt_union), ('fixed', gt_fixed)):
                    c['tp_%s' % ref] = {'0.5': tp, '0.7': tp}
                    c['fp_%s' % ref] = {'0.5': fp, '0.7': fp}
                    c['gtn_%s' % ref] = {'0.5': gt, '0.7': gt}
                configs.append(c)
        configs.sort(key=lambda c: c['k'])
        rows.append({'frame': f, 'scenario': 's%d' % (f // 4), 'N': len(agents), 'ego': 'e',
                     'roster': agents, 'configs': configs})
    return rows


def test_no_drop_means_no_difference_when_references_agree():
    rows = synthetic_rows()
    for r in rows:                         # make the full-roster union equal the fixed reference
        for c in r['configs']:
            if c['k'] == r['N']:
                c['gtn_union'] = dict(c['gtn_fixed'])
    out = resample(rows, M=5, tiers={'none': 0.0})
    for iou in ('0.5', '0.7'):
        cell = out['tiers']['none'][iou]
        assert cell['mean'] == 0.0 and cell['ci'] == [0.0, 0.0]


def test_deterministic_and_drop_rate_monotone():
    rows = synthetic_rows()
    a = resample(rows, M=20)
    b = resample(rows, M=20)
    assert a == b
    kept = [np.mean([len(draw_kept(f, {'e', 'a', 'b'}, 'e', p, rep, t)) for f in range(50) for rep in range(4)])
            for t, p in TIERS.items()]
    assert kept[0] == 3 and kept[0] > kept[1] > kept[2] > kept[3] >= 1


def test_missing_configuration_is_an_error():
    rows = synthetic_rows()
    rows[0]['configs'] = [c for c in rows[0]['configs'] if c['k'] != 1]   # remove ego-only
    rows[0]['configs'][0]['k'] = 99                                      # keep R defined
    with pytest.raises(RuntimeError):
        resample(rows, M=50, tiers={'severe': 0.75})
