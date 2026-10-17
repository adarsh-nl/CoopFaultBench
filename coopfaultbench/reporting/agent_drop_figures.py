# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Agent drop, fixed vs. union reference: one panel, all three models, from
``agent_drop_fixed_vs_union.json``.

x = drop severity (none -> mild -> moderate -> severe, i.e. p_drop 0, 0.25,
0.50, 0.75); y = AP@0.7. Colour = model; solid/filled = fixed reference,
dashed/open = union reference (the standard, agent-dependent one). Each
point is ``models.<m>.tiers.<tier>.0.7.ap_fixed.mean`` /
``...ap_union.mean`` (mean over the M Monte Carlo agent-drop draws). At
'none' both references coincide.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

from . import payloads as pl              # noqa: E402
from . import style as fs                 # noqa: E402

MODELS = ('cobevt', 'v2xvit', 'where2comm')
TIERS = ('none', 'mild', 'moderate', 'severe')
REF_LS = {'fixed': '-', 'union': (0, (3, 1.5))}
REF_MARKER_FILL = {'fixed': 'full', 'union': 'none'}
FIGURE_FILE = 'agent_drop_fixed_vs_union.pdf'


def series(payload, model, ref, iou='0.7', fraction=False):
    """The plotted y values for one (model, reference) curve."""
    key = {'fixed': 'ap_fixed', 'union': 'ap_union'}[ref]
    tiers = payload['models'][model]['tiers']
    norm = (tiers['none'][iou]['ap_fixed']['mean'] if fraction else 1.0)
    return [tiers[t][iou][key]['mean'] / norm for t in TIERS]


def build_fixed_vs_union(payload, fraction=False):
    """fraction=True divides every AP by the model's own 'none'-tier
    AP_fixed (alternative view; the released figure uses absolute AP)."""
    fig, ax = plt.subplots(figsize=(fs.COL_W * 1.55, fs.COL_W * 1.15))
    model_handles = []
    for model in MODELS:
        c = fs.OP_COLOR[model]
        for ref in ('fixed', 'union'):
            means = series(payload, model, ref, fraction=fraction)
            xs = np.arange(len(TIERS))
            ax.plot(xs, means, color=c, linestyle=REF_LS[ref],
                    marker=fs.OP_MARKER[model], markersize=3.2,
                    markerfacecolor=c if REF_MARKER_FILL[ref] == 'full'
                    else 'white', markeredgecolor=c,
                    markeredgewidth=0.6, linewidth=fs.LW['series'])
        model_handles.append(plt.Line2D([0], [0], color=c,
                                        linewidth=fs.LW['series'],
                                        marker=fs.OP_MARKER[model],
                                        markersize=3.2,
                                        label=fs.OP_LABEL[model]))
    ref_handles = [
        plt.Line2D([0], [0], color=fs.REFERENCE_GREY, linestyle=REF_LS['fixed'],
                   linewidth=fs.LW['series'], label='Fixed reference'),
        plt.Line2D([0], [0], color=fs.REFERENCE_GREY, linestyle=REF_LS['union'],
                   linewidth=fs.LW['series'], label='Union reference (as-is)'),
    ]
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([t.capitalize() for t in TIERS], fontsize=fs.FS['tick'])
    ax.set_ylabel('Retained fraction of AP@0.7 (vs.\\ none)' if fraction
                  else 'AP@0.7 (agent drop)', fontsize=fs.FS['label'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=fs.FS['tick'])
    leg1 = ax.legend(handles=model_handles, loc='lower left',
                     fontsize=fs.FS['legend'], handlelength=1.6,
                     title='Model', title_fontsize=fs.FS['legend'],
                     borderaxespad=0.3)
    ax.add_artist(leg1)
    ax.legend(handles=ref_handles, loc='upper right', fontsize=fs.FS['legend'],
              handlelength=1.8, borderaxespad=0.3)
    fig.tight_layout()
    return fig


def render(payload_dir, outdir):
    payload = pl.load_agent_drop(payload_dir)
    os.makedirs(outdir, exist_ok=True)
    with fs.figure_style():
        fig = build_fixed_vs_union(payload)
        p = fs.save_pdf(fig, os.path.join(outdir, FIGURE_FILE))
        plt.close(fig)
    return [p]
