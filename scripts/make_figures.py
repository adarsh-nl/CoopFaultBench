# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Rebuild every released figure (PDF) from the payloads. CPU only.

    python scripts/make_figures.py [--payloads results/payloads] \
        [--out results/figures]

Writes, under ``--out``:
  fault_sweep_gain_forest.pdf, fault_sweep_scene_verdict.pdf,
  fault_sweep_degradation.pdf, fault_sweep_scene_verdict_composition.pdf,
  agent_drop_fixed_vs_union.pdf, agent_drop_by_stratum.pdf,
  severe_tier_ordering.pdf, degradation_all_models.pdf
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use('Agg')

from coopfaultbench.reporting.figures import render_all  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description='Rebuild the released figures.')
    ap.add_argument('--payloads', default=os.path.join(REPO, 'results',
                                                       'payloads'))
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'figures'))
    a = ap.parse_args(argv)
    for p in render_all(a.payloads, a.out):
        print('wrote', os.path.relpath(p))
    return 0


if __name__ == '__main__':
    sys.exit(main())
