# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Fault-sweep figures from ``fault_sweep_fixed_reference.json``.

* ``build_gain_forest``   -- per-condition individual-vs-CP dumbbell, one
  panel per model, AP@0.7; connector solid when the Gain survives
  Holm-Bonferroni (``holm_reject``), dashed and marked 'n.s.' otherwise.
* ``build_scene_verdict`` -- per-scene three-way verdict grid (AP@0.7).
* ``build_degradation``   -- AP@0.7 vs. severity per model x fault family,
  individual (dashed, open) vs. CP (solid, filled) with the 2.5-97.5
  percentile band of the scenario bootstrap.
* ``build_verdict_composition`` -- per (model, fault, severity) fraction of
  the 19 scenes in each verdict, AP@0.5 and AP@0.7 side by side.

Plotted AP values are the payload's bootstrap means (``*_ci[0]``) and
percentile bounds (``*_ci[1]``, ``*_ci[2]``); verdicts are the payload's
per-scene ``verdict`` strings (block ``payloads.KEY_PER_SCENE``). Each
builder must be called inside
``style.figure_style()``; ``render`` does that and writes the PDFs.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from . import payloads as pl              # noqa: E402
from . import style as fs                 # noqa: E402

MODELS = ('cobevt', 'v2xvit', 'where2comm')
INJECTORS = ('pose_error', 'latency', 'points_reduce', 'lidar_fog', 'lidar_snow')
TIERS = ('mild', 'moderate', 'severe')
CONDITIONS = ['clean'] + ['%s_%s' % (f, t) for f in INJECTORS for t in TIERS]

# individual vs CP: same hue as the panel's model, discriminated by
# linestyle/marker fill. CP = solid/filled; individual = dashed/open.
ARM_LS = {'individual': (0, (3, 1.5)), 'cp': '-'}
ARM_MARKER_FILL = {'individual': 'none', 'cp': 'full'}
ARM_LABEL = {'individual': 'Individual (ego-only)', 'cp': 'CP (full roster)'}

# Status palette for the three-way verdict -- fixed, never reused for a
# series, distinct from every model/injector hue.
VERDICT_COLOR = {
    'established beneficial': '#0ca30c',
    'not established': '#BBBBBB',
    'established harmful': '#d03b3b',
    'N/A': '#EEEEEE',
}
VERDICT_ORDER = ['established beneficial', 'not established',
                 'established harmful', 'N/A']

TIER_X = {'clean': 0, 'mild': 1, 'moderate': 2, 'severe': 3}

COMPOSITION_TITLE = ('Per-scene verdict composition -- per severity, left '
                     'bar @0.5 / right bar @0.7')

FIGURE_FILES = {
    'gain_forest': 'fault_sweep_gain_forest.pdf',
    'scene_verdict': 'fault_sweep_scene_verdict.pdf',
    'degradation': 'fault_sweep_degradation.pdf',
    'verdict_composition': 'fault_sweep_scene_verdict_composition.pdf',
}


def band(ax, x, lo, hi, color, alpha=0.18):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def cond_label(cond):
    if cond == 'clean':
        return 'Clean'
    fault, tier = cond.rsplit('_', 1)
    return '%s %s' % (fs.INJ_LABEL.get(fault, fault), tier)


def build_gain_forest(payload, iou='0.7'):
    fig, axes = plt.subplots(1, 3, figsize=(fs.FULL_W, fs.FULL_W * 0.95),
                             sharey=True, sharex=True)
    y = np.arange(len(CONDITIONS))[::-1]
    for ax, model in zip(axes, MODELS):
        per_cond = payload['models'][model][pl.KEY_PER_CONDITION][iou]
        for yi, cond in zip(y, CONDITIONS):
            d = per_cond.get(cond)
            if d is None or d.get('ap_indiv_ci') is None \
                    or d.get('ap_cp_ci') is None:
                ax.text(0.5, yi, 'N/A', fontsize=fs.FS['annot'],
                        color=fs.REFERENCE_GREY, va='center', ha='center',
                        transform=ax.get_yaxis_transform())
                continue
            smean, slo, shi = d['ap_indiv_ci']
            fmean, flo, fhi = d['ap_cp_ci']
            color = (fs.REFERENCE_GREY if cond == 'clean'
                     else fs.INJ_COLOR[cond.rsplit('_', 1)[0]])
            sig = bool(d.get('holm_reject'))
            ax.plot([smean, fmean], [yi, yi], color=color,
                    linewidth=fs.LW['series'] if sig else fs.LW['thin'],
                    linestyle='-' if sig else (0, (2, 1.5)),
                    alpha=1.0 if sig else 0.55, zorder=2)
            for mean, lo, hi, fill in ((smean, slo, shi, 'none'),
                                       (fmean, flo, fhi, color)):
                ax.plot([lo, hi], [yi, yi], color=color,
                        linewidth=fs.LW['thin'], alpha=0.6, zorder=1)
                ax.scatter([mean], [yi], color=fill, marker='o', s=13,
                           zorder=3, edgecolors=color, linewidths=0.7)
            if not sig:
                # Clear the CI whiskers, not just the point estimates.
                ax.text(max(shi, fhi) + 0.015, yi, 'n.s.',
                        fontsize=fs.FS['annot'], color=fs.REFERENCE_GREY,
                        va='center', ha='left')
        ax.set_title(fs.OP_LABEL[model], fontsize=fs.FS['panel'], pad=2,
                     loc='left')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(left=False, labelsize=fs.FS['tick'])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([cond_label(c) for c in CONDITIONS],
                            fontsize=fs.FS['tick'])
    fig.supxlabel('AP @%s -- solo (open) $\\rightarrow$ CP (filled)' % iou,
                  fontsize=fs.FS['label'])
    fig.tight_layout(w_pad=0.8, rect=[0, 0.04, 1, 1])
    return fig


def scene_list(payload, iou='0.7'):
    return sorted({s for model in MODELS
                   for s in payload['models'][model][pl.KEY_PER_SCENE][iou]
                   .get(CONDITIONS[0], {})})


def verdict_grid(payload, model, scenes, iou='0.7'):
    """(n_scenes, n_conditions) array of VERDICT_ORDER indices."""
    verdicts = payload['models'][model][pl.KEY_PER_SCENE][iou]
    grid = np.zeros((len(scenes), len(CONDITIONS)))
    for ci, cond in enumerate(CONDITIONS):
        per_scene = verdicts.get(cond, {})
        for si, scen in enumerate(scenes):
            v = per_scene.get(scen, {}).get('verdict', 'N/A')
            grid[si, ci] = VERDICT_ORDER.index(v)
    return grid


def build_scene_verdict(payload, iou='0.7'):
    scenes = scene_list(payload, iou)
    fig, axes = plt.subplots(1, 3, figsize=(fs.FULL_W, fs.FULL_W * 0.62),
                             sharey=True)
    for ax, model in zip(axes, MODELS):
        grid = verdict_grid(payload, model, scenes, iou)
        cmap = ListedColormap([VERDICT_COLOR[v] for v in VERDICT_ORDER])
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(VERDICT_ORDER) - 0.5,
                  aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([cond_label(c) for c in CONDITIONS], rotation=90,
                           fontsize=fs.FS['small'])
        ax.set_title(fs.OP_LABEL[model], fontsize=fs.FS['panel'], pad=2,
                     loc='left')
        ax.set_xticks(np.arange(-0.5, len(CONDITIONS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(scenes), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.6)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.tick_params(which='major', bottom=False, left=False)
    axes[0].set_yticks(range(len(scenes)))
    axes[0].set_yticklabels(scenes, fontsize=fs.FS['small'])
    handles = [plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR[v])
               for v in VERDICT_ORDER]
    fig.legend(handles, VERDICT_ORDER, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=fs.FS['legend'],
               frameon=False, columnspacing=1.2)
    fig.tight_layout(w_pad=0.8)
    return fig


def build_degradation(payload, iou='0.7'):
    """Pending cells (none in the released payload) are never interpolated
    through or zeroed: the line breaks there and 'pending' is printed."""
    tiers = ('clean', 'mild', 'moderate', 'severe')
    rows_all = pl.all_rows(payload)
    fig, axes = plt.subplots(len(MODELS), len(INJECTORS),
                             figsize=(fs.FULL_W, fs.FULL_W * 0.82),
                             sharex=True, sharey='row')
    for mi, model in enumerate(MODELS):
        c = fs.OP_COLOR[model]
        for fi, fault in enumerate(INJECTORS):
            ax = axes[mi, fi]
            any_pending = False
            for arm, key in (('individual', 'ap_indiv_ci'),
                             ('cp', 'ap_cp_ci')):
                xs, means, los, his = [], [], [], []
                for tier in tiers:
                    cond = ('clean' if tier == 'clean'
                            else '%s_%s' % (fault, tier))
                    r = rows_all[(model, cond, iou)]
                    if r['pending']:
                        any_pending = True
                        continue
                    xs.append(TIER_X[tier])
                    m, lo, hi = r[key]
                    means.append(m)
                    los.append(lo)
                    his.append(hi)
                if xs:
                    band(ax, xs, los, his, c)
                    ax.plot(xs, means, color=c, linestyle=ARM_LS[arm],
                            marker=fs.OP_MARKER[model], markersize=2.6,
                            markerfacecolor=c if ARM_MARKER_FILL[arm] == 'full'
                            else 'white', markeredgecolor=c,
                            markeredgewidth=0.5, linewidth=fs.LW['series'])
            if any_pending:
                ylim = ax.get_ylim()
                for tier in tiers:
                    cond = ('clean' if tier == 'clean'
                            else '%s_%s' % (fault, tier))
                    if rows_all[(model, cond, iou)]['pending']:
                        xi = TIER_X[tier]
                        ax.axvline(xi, color=fs.GRID_GREY,
                                   linewidth=fs.LW['thin'],
                                   linestyle=(0, (1, 1)), zorder=0)
                        ax.text(xi, ylim[1], 'pending', rotation=90,
                                fontsize=fs.FS['annot'],
                                color=fs.REFERENCE_GREY,
                                va='top', ha='center')
            if mi == 0:
                ax.set_title(fs.INJ_LABEL[fault], fontsize=fs.FS['panel'],
                             pad=2)
            if fi == 0:
                ax.set_ylabel('%s\nAP@%s' % (fs.OP_LABEL[model], iou),
                              fontsize=fs.FS['small'])
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=fs.FS['tick'])
    for ax in axes[-1]:
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels([t.capitalize() for t in tiers],
                           fontsize=fs.FS['small'], rotation=45, ha='right')
    arm_handles = [
        plt.Line2D([0], [0], color=fs.REFERENCE_GREY,
                   linestyle=ARM_LS['individual'],
                   linewidth=fs.LW['series'], marker='o', markersize=2.6,
                   markerfacecolor='white', markeredgecolor=fs.REFERENCE_GREY,
                   label=ARM_LABEL['individual']),
        plt.Line2D([0], [0], color=fs.REFERENCE_GREY, linestyle=ARM_LS['cp'],
                   linewidth=fs.LW['series'], marker='o', markersize=2.6,
                   markerfacecolor=fs.REFERENCE_GREY,
                   markeredgecolor=fs.REFERENCE_GREY,
                   label=ARM_LABEL['cp']),
    ]
    fig.legend(handles=arm_handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.045), fontsize=fs.FS['legend'],
               frameon=False, columnspacing=1.4, handlelength=2.2)
    fig.suptitle('AP@%s vs.\\ severity, individual vs.\\ CP -- rows = model '
                 '(colour), columns = fault family' % iou,
                 fontsize=fs.FS['title'], y=1.01)
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=[0, 0.03, 1, 1])
    return fig


def verdict_fractions(payload, model, cond, iou):
    """{verdict: fraction of scenes} for one (model, condition, iou), or
    None when the condition is halted or absent."""
    cell = payload['models'][model][pl.KEY_PER_SCENE].get(iou, {}).get(cond)
    halted = (model, cond) in pl.halted_lookup(payload)
    if halted or not cell:
        return None
    counts = {v: 0 for v in VERDICT_ORDER}
    for scen_d in cell.values():
        counts[scen_d.get('verdict', 'N/A')] += 1
    total = sum(counts.values()) or 1
    return {v: counts[v] / total for v in VERDICT_ORDER}


def build_verdict_composition(payload):
    fig, axes = plt.subplots(len(MODELS), len(INJECTORS),
                             figsize=(fs.FULL_W, fs.FULL_W * 0.82),
                             sharex=True, sharey=True)
    bar_w = 0.34
    for mi, model in enumerate(MODELS):
        for fi, fault in enumerate(INJECTORS):
            ax = axes[mi, fi]
            for ti, tier in enumerate(TIERS):
                cond = '%s_%s' % (fault, tier)
                for ii, iou in enumerate(pl.IOUS):
                    xoff = ti + (ii - 0.5) * bar_w
                    fr = verdict_fractions(payload, model, cond, iou)
                    if fr is None:
                        ax.text(xoff, 0.5, 'pending', rotation=90,
                                fontsize=fs.FS['annot'],
                                color=fs.REFERENCE_GREY,
                                va='center', ha='center')
                        continue
                    bottom = 0.0
                    for v in VERDICT_ORDER:
                        frac = fr[v]
                        if frac <= 0:
                            continue
                        ax.bar(xoff, frac, width=bar_w, bottom=bottom,
                               color=VERDICT_COLOR[v], edgecolor='white',
                               linewidth=0.3)
                        bottom += frac
            if mi == 0:
                ax.set_title(fs.INJ_LABEL[fault], fontsize=fs.FS['panel'],
                             pad=2)
            if fi == 0:
                ax.set_ylabel(fs.OP_LABEL[model], fontsize=fs.FS['small'])
            ax.set_ylim(0, 1.02)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=fs.FS['tick'])
    for ax in axes[-1]:
        ax.set_xticks(range(len(TIERS)))
        ax.set_xticklabels([t.capitalize() for t in TIERS],
                           fontsize=fs.FS['small'], rotation=45, ha='right')
    axes[0, 0].set_ylabel(axes[0, 0].get_ylabel() + '\nFraction of 19 scenes',
                          fontsize=fs.FS['small'])
    verdict_handles = [plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR[v])
                       for v in VERDICT_ORDER]
    iou_note = plt.Line2D([0], [0], color='none',
                          label='each severity: left bar @0.5, right bar @0.7')
    fig.legend(handles=verdict_handles + [iou_note],
               labels=VERDICT_ORDER + [iou_note.get_label()],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.055),
               fontsize=fs.FS['legend'], frameon=False, columnspacing=1.2)
    fig.suptitle(COMPOSITION_TITLE, fontsize=fs.FS['title'], y=1.01)
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=[0, 0.04, 1, 1])
    return fig


BUILDERS = (
    ('gain_forest', build_gain_forest),
    ('scene_verdict', build_scene_verdict),
    ('degradation', build_degradation),
    ('verdict_composition', build_verdict_composition),
)


def render(payload_dir, outdir):
    payload = pl.load_fault_sweep(payload_dir)
    os.makedirs(outdir, exist_ok=True)
    written = []
    with fs.figure_style():
        for key, builder in BUILDERS:
            fig = builder(payload)
            written.append(fs.save_pdf(
                fig, os.path.join(outdir, FIGURE_FILES[key])))
            plt.close(fig)
    return written
