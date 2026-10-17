# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""CPU-only table and figure builders for the released ``results/`` tree.

Every builder reads only the files under ``results/payloads`` (numpy,
matplotlib and the standard library; no GPU, no raw data, no model weights).
Entry points: ``scripts/make_tables.py`` and ``scripts/make_figures.py``.
"""
from __future__ import annotations

# Public file names inside a payload directory.
FAULT_SWEEP_PAYLOAD = 'fault_sweep_fixed_reference.json'
AGENT_DROP_PAYLOAD = 'agent_drop_fixed_vs_union.json'
GRID_CSV = 'fault_inference_grid.csv'
GRID_MODELS_JSON = 'fault_inference_models.json'
GRID_CELLS_DIR = 'cells'
