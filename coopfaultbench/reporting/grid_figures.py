# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Figures from the 109-row fault-inference grid (``fault_inference_grid.csv``).

* ``build_agent_drop_by_stratum`` -- agent drop (union reference) per scene
  agent count; top row bars from zero, bottom row the same data zoomed;
  per-stratum clean reference in orange. CoSDH cannot appear (no agent-drop
  cells).
* ``build_severe_tier_ordering`` -- severe-tier relative AP@0.7 drop by
  injector and model, grouped into corrupted-data and lost-collaborator
  bands; agent drop is the n=2 stratum only and carries a dagger (its
  reference set moves with the fault).
* ``build_degradation_all_models`` -- absolute AP@0.7 vs. severity per
  injector for all four models, with the clean AP as a dashed line.

Values plotted: grid columns ``faulty_ap70``, ``clean_ap70``,
``rel_drop_ap70``. Builders must be called inside
``style.figure_style(style.GRID_FAMILY_RC)``; ``render`` does that.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402

from . import payloads as pl                                     # noqa: E402
from . import style                                              # noqa: E402
from .style import (COL_W, FULL_W, INJ_COLOR, INJ_LABEL,         # noqa: E402
                    OP_LABEL, inj_style)

CORRUPTED = ['lidar_snow', 'lidar_fog', 'points_reduce', 'pose_error',
             'latency']
LOST = ['missing_modality', 'agent_drop']
TIERS = ['mild', 'moderate', 'severe']
MODELS = ['cobevt', 'v2xvit', 'where2comm', 'cosdh']
LABEL = OP_LABEL

FIGURE_FILES = {
    'agent_drop_by_stratum': 'agent_drop_by_stratum.pdf',
    'severe_tier_ordering': 'severe_tier_ordering.pdf',
    'degradation_all_models': 'degradation_all_models.pdf',
}


def one(rows, **kw):
    h = [r for r in rows if all(str(r.get(k)) == str(v)
                                for k, v in kw.items())]
    return h[0] if len(h) == 1 else None


def build_agent_drop_by_stratum(rows):
    ops = [m for m in MODELS if any(r['model'] == m
                                    and r['injector'] == 'agent_drop'
                                    for r in rows)]
    fig, axes = plt.subplots(2, len(ops), figsize=(FULL_W, 3.5),
                             gridspec_kw={'height_ratios': [1.0, 0.85],
                                          'hspace': 0.30})
    alpha = {'mild': 0.45, 'moderate': 0.72, 'severe': 1.0}
    for col, m in enumerate(ops):
        ax, axz = axes[0][col], axes[1][col]
        strata = sorted({int(r['_x']['scene_agent_count']) for r in rows
                         if r['model'] == m and r['injector'] == 'agent_drop'})
        w, lo, hi = 0.26, 1e9, -1e9
        for si, n in enumerate(strata):
            for ti, t in enumerate(TIERS):
                h = [r for r in rows if r['model'] == m
                     and r['injector'] == 'agent_drop'
                     and r['severity_tier'] == t
                     and r['_x'].get('scene_agent_count') == str(n)]
                v = h[0]['faulty_ap70']
                lo, hi = min(lo, v), max(hi, v)
                for a in (ax, axz):
                    a.bar(si + (ti - 1) * w, v, width=w,
                          color=INJ_COLOR['agent_drop'], alpha=alpha[t],
                          edgecolor='#333333', lw=0.4, zorder=2)
            cl = h[0]['clean_ap70']
            lo, hi = min(lo, cl), max(hi, cl)
            for a in (ax, axz):
                a.plot([si - 1.6 * w, si + 1.6 * w], [cl, cl],
                       color='#D55E00', lw=1.0, zorder=3)
                a.plot([si], [cl], marker='o', ms=3, color='#D55E00',
                       zorder=4)
        for a in (ax, axz):
            a.set_xticks(range(len(strata)))
            a.set_xticklabels(['%d' % n for n in strata])
            a.grid(axis='y', color='#dddddd', zorder=0)
        ax.set_title('%s' % LABEL[m], fontsize=8)
        axz.set_ylim(max(0.0, lo - 0.02), hi + 0.02)
        axz.set_title('zoom %.2f-%.2f' % axz.get_ylim(), fontsize=7,
                      loc='left', color='0.35', pad=2)
        # One x label, on the middle column only.
        if col == len(ops) // 2:
            axz.set_xlabel('scene agent count', fontsize=7.5)
    axes[0][0].set_ylabel('AP@0.7 (from 0)', fontsize=7.5)
    axes[1][0].set_ylabel('AP@0.7 (zoom)', fontsize=7.5)
    hs = [Patch(facecolor=INJ_COLOR['agent_drop'], alpha=alpha[t],
                edgecolor='#333333', lw=0.4, label=t.capitalize())
          for t in TIERS]
    hs.append(Line2D([], [], color='#D55E00', marker='o',
                     label='Per-stratum clean'))
    fig.legend(handles=hs, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.09))
    return fig


def severe_value(rows, m, i):
    """Plotted bar length: severe-tier ``rel_drop_ap70`` (agent drop: the
    scene_agent_count=2 stratum). NaN (not drawn) for CoSDH's absent
    lost-collaborator cells."""
    if m == 'cosdh' and i in LOST:
        return float('nan')
    if i == 'agent_drop':
        h = [r for r in rows if r['model'] == m
             and r['injector'] == 'agent_drop'
             and r['severity_tier'] == 'severe'
             and r['_x'].get('scene_agent_count') == '2']
        return h[0]['rel_drop_ap70']
    r = one(rows, model=m, injector=i, severity_tier='severe')
    return r['rel_drop_ap70'] if r else float('nan')


def severe_order(rows):
    def key(i):
        v = [severe_value(rows, m, i) for m in MODELS]
        v = [x for x in v if x == x]
        return sum(v) / len(v)
    return sorted(CORRUPTED + LOST, key=key)


def build_severe_tier_ordering(rows):
    order = severe_order(rows)
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    h, off = 0.19, [-0.285, -0.095, 0.095, 0.285]
    alphas = [1.0, 0.78, 0.56, 0.38]
    hatch = {'where2comm': '///', 'cosdh': 'xx'}
    lost_rows = [k for k, i in enumerate(order) if i in LOST]
    # Band grouping made visually explicit.
    ax.axhspan(min(lost_rows) - 0.5, max(lost_rows) + 0.5, color='#ececec',
               zorder=0)
    for k, m in enumerate(MODELS):
        for r_, inj in enumerate(order):
            v = severe_value(rows, m, inj)
            if v != v:
                continue
            ax.barh(r_ + off[k], v, height=h, color=INJ_COLOR[inj],
                    alpha=alphas[k], edgecolor='#333333', lw=0.4,
                    hatch=hatch.get(m, ''))
    # Divider between the two bands.
    ax.axhline(min(lost_rows) - 0.5, color='#555555', lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(range(len(order)))
    # Agent drop keeps its moving-reference flag.
    ax.set_yticklabels([INJ_LABEL[i] + (r' $^{\dagger}$'
                                        if i == 'agent_drop' else '')
                        for i in order])
    ax.text(0.02, (min(lost_rows) + max(lost_rows)) / 2.0,
            'lost collaborators', transform=ax.get_yaxis_transform(),
            fontsize=6.5, va='center', color='#555555')
    corr_rows = [k for k, i in enumerate(order) if i not in LOST]
    ax.text(0.02, (min(corr_rows) + max(corr_rows)) / 2.0, 'corrupted data',
            transform=ax.get_yaxis_transform(), fontsize=6.5, va='center',
            color='#555555')
    ax.set_xlabel('Relative AP@0.7 drop (severe tier)', fontsize=7.5)
    ax.set_title('Fault severity ranking, severe tier', fontsize=7.5,
                 loc='left', pad=4)
    ax.grid(axis='x', color='#dddddd', zorder=0)
    hs = [Patch(facecolor='#888888', alpha=a, hatch=hatch.get(m, ''),
                edgecolor='#333333', lw=0.4, label=LABEL[m])
          for m, a in zip(MODELS, alphas)]
    ax.legend(handles=hs, fontsize=6.5, ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, -0.20), columnspacing=1.0,
              handlelength=1.3)
    return fig


def build_degradation_all_models(rows):
    injs = CORRUPTED + ['missing_modality']
    fig, axes = plt.subplots(1, len(MODELS), figsize=(FULL_W, 2.0),
                             sharex=True)
    for ax, m in zip(axes, MODELS):
        c70 = one(rows, model=m, injector='none',
                  severity_tier='clean')['clean_ap70']
        ax.axhline(c70, ls=(0, (4, 2)), lw=0.8, color='#222222', zorder=1)
        for inj in injs:
            if m == 'cosdh' and inj in LOST:
                continue
            ys = [one(rows, model=m, injector=inj,
                      severity_tier=t)['faulty_ap70'] for t in TIERS]
            ax.plot(range(3), ys, zorder=3,
                    **inj_style(inj, markersize=2.8, linewidth=1.1,
                                label=None))
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Mild', 'Mod', 'Sev'], fontsize=6.5)
        ax.set_title(LABEL[m], fontsize=8)
        ax.grid(axis='y', color='#dddddd', zorder=0)
    axes[0].set_ylabel('AP@0.7', fontsize=7.5)
    hs = [Line2D([], [], **inj_style(i, markersize=3.0))
          for i in injs]
    hs.append(Line2D([], [], color='#222222', ls=(0, (4, 2)), label='Clean'))
    fig.legend(handles=hs, loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 0.01), fontsize=6.5, columnspacing=1.0)
    return fig


BUILDERS = (
    ('agent_drop_by_stratum', build_agent_drop_by_stratum),
    ('severe_tier_ordering', build_severe_tier_ordering),
    ('degradation_all_models', build_degradation_all_models),
)


def render(payload_dir, outdir):
    rows = pl.load_grid_rows_float(payload_dir)
    os.makedirs(outdir, exist_ok=True)
    written = []
    with style.figure_style(style.GRID_FAMILY_RC):
        for key, builder in BUILDERS:
            fig = builder(rows)
            written.append(style.save_pdf(
                fig, os.path.join(outdir, FIGURE_FILES[key])))
            plt.close(fig)
    return written
