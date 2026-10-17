# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Build every released table from a payload directory.

``build_all_tables(payload_dir)`` returns ``{file name: bytes}``; nothing is
written. ``write_tables`` writes them. The file names are the public names
under ``results/tables``.
"""
from __future__ import annotations

import os

from . import agent_drop_tables, fault_sweep_tables, grid_tables
from . import payloads as pl

TEXT_ENCODING = 'utf-8'


def _b(text):
    return text.encode(TEXT_ENCODING)


def build_fault_sweep_tables(payload_dir):
    payload = pl.load_fault_sweep(payload_dir)
    out = {}
    for model in pl.MODELS:
        tex, _md = fault_sweep_tables.build_model_table(payload, model)
        out['fault_sweep_%s.tex' % model] = _b(tex + '\n')
        out['fault_sweep_%s.csv' % model] = _b(
            fault_sweep_tables.build_csv(payload, model))
    out['fault_sweep_tables.md'] = _b(
        fault_sweep_tables.build_markdown(payload))
    return out


def build_agent_drop_tables(payload_dir):
    payload = pl.load_agent_drop(payload_dir)
    return {
        'agent_drop_fixed_vs_union.tex': _b(
            agent_drop_tables.build_tex(payload)),
        'agent_drop_fixed_vs_union.md': _b(
            agent_drop_tables.build_markdown(payload)),
        'agent_drop_fixed_vs_union.csv': _b(
            agent_drop_tables.build_csv(payload)),
    }


def build_grid_tables(payload_dir):
    rows = pl.load_grid_rows(payload_dir)
    models = pl.load_grid_models(payload_dir)

    def cell_loader(model, injector, tier):
        return pl.load_grid_cell(payload_dir, model, injector, tier)

    return {
        'model_reproduction.tex': _b(grid_tables.build_model_reproduction(
            rows, models, cell_loader)),
        'severe_tier_ordering.tex': _b(
            grid_tables.build_severe_tier_ordering(rows)),
        'agent_drop_by_stratum.tex': _b(
            grid_tables.build_agent_drop_by_stratum(rows)),
    }


def build_all_tables(payload_dir):
    out = {}
    for part in (build_fault_sweep_tables(payload_dir),
                 build_agent_drop_tables(payload_dir),
                 build_grid_tables(payload_dir)):
        for k, v in part.items():
            if k in out:
                raise ValueError('duplicate table name %s' % k)
            out[k] = v
    return out


def write_tables(payload_dir, outdir):
    os.makedirs(outdir, exist_ok=True)
    written = []
    for name, data in sorted(build_all_tables(payload_dir).items()):
        p = os.path.join(outdir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        written.append(p)
    return written
