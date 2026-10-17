#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Monte Carlo agent drop over agent-subset rows -> AP_union, AP_fixed, dAP per model, drop rate and IoU.

CPU only, no OpenCOOD. Input: one directory per model holding rows_*.jsonl from
scripts/run_agent_subset_sweep.py. Output: a JSON payload with the structure of
results/payloads/agent_drop_fixed_vs_union.json (see docs/results_schema.md).

    python scripts/resample_agent_drop.py --rows_dir SWEEP_DIR --models cobevt,v2xvit,where2comm \
        --out agent_drop.json [--compare results/payloads/agent_drop_fixed_vs_union.json]

With --compare, every value is compared with the given payload and the script exits 1 on any difference.
The paper's payload was produced from the same rows with M = 100 (2,834 frames per model). The rows
themselves are not distributed (they are derived per frame from the dataset).
"""
import argparse
import json
import os
import sys

from coopfaultbench.evaluation.agent_drop import TIERS, load_rows, resample


def compare(a, b, path=''):
    if isinstance(b, dict):
        out = [] if set(a) == set(b) else ['%s: keys differ' % path]
        for k in b:
            if k in a:
                out += compare(a[k], b[k], '%s.%s' % (path, k))
        return out
    if isinstance(b, list):
        head = [] if len(a) == len(b) else ['%s: length %d != %d' % (path, len(a), len(b))]
        return head + [d for i, (x, y) in enumerate(zip(a, b)) for d in compare(x, y, '%s[%d]' % (path, i))]
    return [] if a == b else ['%s: %r != %r' % (path, a, b)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rows_dir', required=True, help='directory with one sub-directory of rows per model')
    ap.add_argument('--models', default='cobevt,v2xvit,where2comm')
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--out', required=True)
    ap.add_argument('--compare', default=None)
    a = ap.parse_args()
    payload = {'M': a.M, 'seed': 20260826, 'tiers': dict(TIERS), 'models': {}}
    for m in [x.strip() for x in a.models.split(',') if x.strip()]:
        payload['models'][m] = resample(load_rows(os.path.join(a.rows_dir, m)), M=a.M)
        print('%s: %d frames' % (m, payload['models'][m]['n_frames']), flush=True)
    with open(a.out, 'w') as fh:
        json.dump(payload, fh, indent=1)
    print('wrote', a.out)
    if a.compare:
        diffs = compare(payload, json.load(open(a.compare)))
        print('%d difference(s) against %s' % (len(diffs), a.compare))
        for d in diffs[:20]:
            print('  ', d)
        return 1 if diffs else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
