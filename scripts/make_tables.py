# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Rebuild every released table from the payloads. CPU only.

    python scripts/make_tables.py [--payloads results/payloads] \
        [--out results/tables]

Writes, under ``--out``:
  fault_sweep_<model>.tex / .csv, fault_sweep_tables.md
  agent_drop_fixed_vs_union.tex / .md / .csv
  model_reproduction.tex, severe_tier_ordering.tex, agent_drop_by_stratum.tex
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from coopfaultbench.reporting.tables import write_tables  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description='Rebuild the released tables.')
    ap.add_argument('--payloads', default=os.path.join(REPO, 'results',
                                                       'payloads'))
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'tables'))
    a = ap.parse_args(argv)
    for p in write_tables(a.payloads, a.out):
        print('wrote', os.path.relpath(p))
    return 0


if __name__ == '__main__':
    sys.exit(main())
