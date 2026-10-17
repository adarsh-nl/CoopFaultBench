# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Render every released figure (PDF) from a payload directory."""
from __future__ import annotations

from . import agent_drop_figures, fault_sweep_figures, grid_figures

FAMILIES = (
    ('fault sweep', fault_sweep_figures),
    ('agent drop', agent_drop_figures),
    ('inference grid', grid_figures),
)


def expected_files():
    out = list(fault_sweep_figures.FIGURE_FILES.values())
    out.append(agent_drop_figures.FIGURE_FILE)
    out += list(grid_figures.FIGURE_FILES.values())
    return sorted(out)


def render_all(payload_dir, outdir):
    written = []
    for _name, mod in FAMILIES:
        written += mod.render(payload_dir, outdir)
    return written
