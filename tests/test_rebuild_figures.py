# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Figure rebuild checks.

* ``scripts/make_figures.py`` runs and writes every expected, non-empty PDF.
* With the matplotlib and fontTools versions that produced the shipped PDFs,
  regeneration has identical CONTENT to ``results/figures/*``: every object,
  every decompressed stream and every image pixel (``tests/pdf_content.py``).
  The compressed bytes of image streams can differ across CPU architectures
  (Pillow's bundled zlib), so the files are not compared byte for byte.
  Skipped on other matplotlib/fontTools versions, where content differs.
* Controls: recompressing a PDF keeps its content; one changed pixel, one
  changed label glyph or a changed payload value changes it.
* The plotted data -- read back from the matplotlib artists -- equal the
  payload values, and a perturbed payload moves the plotted data.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from coopfaultbench.reporting import (agent_drop_figures,  # noqa: E402
                                      fault_sweep_figures, grid_figures)
from coopfaultbench.reporting import payloads as pl  # noqa: E402
from coopfaultbench.reporting import style  # noqa: E402
from coopfaultbench.reporting.figures import expected_files  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdf_content  # noqa: E402

PAYLOADS = os.path.join(REPO, 'results', 'payloads')
FIGURES = os.path.join(REPO, 'results', 'figures')
# matplotlib version that wrote the shipped PDFs (recorded in their
# /Creator field); content equality is only meaningful with the same version.
SHIPPED_MPL = '3.11.1'
SHIPPED_FONTTOOLS = '4.63.0'  # Type-42 font subsetting

EXPECTED = sorted([
    'fault_sweep_gain_forest.pdf', 'fault_sweep_scene_verdict.pdf',
    'fault_sweep_degradation.pdf',
    'fault_sweep_scene_verdict_composition.pdf',
    'agent_drop_fixed_vs_union.pdf', 'agent_drop_by_stratum.pdf',
    'severe_tier_ordering.pdf', 'degradation_all_models.pdf',
])


@pytest.fixture(scope='module')
def rendered(tmp_path_factory):
    out = str(tmp_path_factory.mktemp('figures'))
    env = dict(os.environ)
    env.setdefault('MPLBACKEND', 'Agg')
    res = subprocess.run([sys.executable, '-B',
                          os.path.join(REPO, 'scripts', 'make_figures.py'),
                          '--payloads', PAYLOADS, '--out', out],
                         env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    assert res.returncode == 0, res.stderr.decode()
    return out


def test_expected_figure_set(rendered):
    assert expected_files() == EXPECTED
    assert sorted(os.listdir(rendered)) == EXPECTED
    assert sorted(os.listdir(FIGURES)) == EXPECTED


def test_each_figure_is_a_nonempty_pdf(rendered):
    for name in EXPECTED:
        with open(os.path.join(rendered, name), 'rb') as fh:
            data = fh.read()
        assert data.startswith(b'%PDF-') and len(data) > 5000, name
        assert b'CreationDate' not in data, name


def _same_toolchain():
    try:
        import fontTools
        ft = fontTools.version
    except Exception:  # pragma: no cover
        ft = None
    return matplotlib.__version__ == SHIPPED_MPL and ft == SHIPPED_FONTTOOLS


@pytest.mark.skipif(not _same_toolchain(),
                    reason='PDF content depends on the matplotlib/fontTools '
                           'versions')
def test_figures_content_identical_to_shipped(rendered):
    n = 0
    for name in EXPECTED:
        with open(os.path.join(rendered, name), 'rb') as a, \
                open(os.path.join(FIGURES, name), 'rb') as b:
            assert pdf_content.content(a.read()) == pdf_content.content(b.read()), name
        n += 1
    assert n == 8


def _shipped(name):
    with open(os.path.join(FIGURES, name), 'rb') as fh:
        return fh.read()


def test_content_ignores_compression():
    """What a different zlib (another CPU architecture) does: other bytes, same content."""
    for name in EXPECTED:
        data = _shipped(name)
        other = pdf_content.rewrite_streams(data, level=1)
        assert other != data, name
        assert pdf_content.content(other) == pdf_content.content(data), name


def test_content_detects_one_changed_pixel():
    data = _shipped('fault_sweep_scene_verdict.pdf')
    hit = []

    def flip(num, head, dec):
        if b'/Subtype /Image' in head and not hit:
            hit.append(num)
            b = bytearray(dec)
            b[1 + 37] ^= 0x40              # row 0 (after its filter byte): one 2-bit pixel index
            return bytes(b)
        return dec
    other = pdf_content.rewrite_streams(data, edit=flip)
    assert hit
    assert pdf_content.content(other) != pdf_content.content(data)


def test_content_detects_one_changed_label_glyph():
    data = _shipped('agent_drop_fixed_vs_union.pdf')
    hit = []

    def relabel(num, head, dec):
        if not hit and b'/Subtype' not in head and b'TJ' in dec:
            i = dec.index(b'TJ')
            j = dec.rindex(b'(', 0, i) + 1      # first byte of the last string before that TJ
            hit.append(num)
            return dec[:j] + bytes([dec[j] ^ 1]) + dec[j + 1:]
        return dec
    other = pdf_content.rewrite_streams(data, edit=relabel)
    assert hit
    assert pdf_content.content(other) != pdf_content.content(data)


# ------------------------------------------------------ plotted == payload
def _lines(ax):
    return [ln for ln in ax.get_lines()]


def test_fault_sweep_degradation_plots_payload_means():
    payload = pl.load_fault_sweep(PAYLOADS)
    with style.figure_style():
        fig = fault_sweep_figures.build_degradation(payload)
        axes = np.array(fig.axes[:15]).reshape(3, 5)
        n = 0
        for mi, model in enumerate(fault_sweep_figures.MODELS):
            for fi, fault in enumerate(fault_sweep_figures.INJECTORS):
                lines = _lines(axes[mi, fi])
                assert len(lines) == 2
                conds = ['clean'] + ['%s_%s' % (fault, t)
                                     for t in ('mild', 'moderate', 'severe')]
                q = payload['models'][model][pl.KEY_PER_CONDITION]['0.7']
                for ln, key in zip(lines, ('ap_indiv_ci', 'ap_cp_ci')):
                    assert list(ln.get_xdata()) == [0, 1, 2, 3]
                    assert list(ln.get_ydata()) == [q[c][key][0]
                                                    for c in conds]
                    n += 1
        plt.close(fig)
    assert n == 30


def test_fault_sweep_forest_plots_payload_means():
    payload = pl.load_fault_sweep(PAYLOADS)
    with style.figure_style():
        fig = fault_sweep_figures.build_gain_forest(payload)
        for ax, model in zip(fig.axes, fault_sweep_figures.MODELS):
            pts = [tuple(c.get_offsets()[0]) for c in ax.collections
                   if isinstance(c, PathCollection)]
            assert len(pts) == 2 * 16
            q = payload['models'][model][pl.KEY_PER_CONDITION]['0.7']
            ys = list(range(15, -1, -1))
            want = []
            for y, cond in zip(ys, fault_sweep_figures.CONDITIONS):
                want += [(q[cond]['ap_indiv_ci'][0], y),
                         (q[cond]['ap_cp_ci'][0], y)]
            assert pts == want
        plt.close(fig)


def test_fault_sweep_scene_verdict_grid_is_payload_verdicts():
    payload = pl.load_fault_sweep(PAYLOADS)
    order = fault_sweep_figures.VERDICT_ORDER
    with style.figure_style():
        fig = fault_sweep_figures.build_scene_verdict(payload)
        for ax, model in zip(fig.axes, fault_sweep_figures.MODELS):
            arr = np.asarray(ax.get_images()[0].get_array())
            per_scene = payload['models'][model][pl.KEY_PER_SCENE]['0.7']
            scenes = sorted(per_scene['clean'])
            assert arr.shape == (19, 16)
            for ci, cond in enumerate(fault_sweep_figures.CONDITIONS):
                for si, s in enumerate(scenes):
                    assert arr[si, ci] == order.index(
                        per_scene[cond][s]['verdict'])
        plt.close(fig)


def test_fault_sweep_composition_bars_are_scene_fractions():
    payload = pl.load_fault_sweep(PAYLOADS)
    with style.figure_style():
        fig = fault_sweep_figures.build_verdict_composition(payload)
        ax = fig.axes[0]  # CoBEVT, pose error
        bars = [p for p in ax.patches if isinstance(p, Rectangle)]
        got = {}
        for b in bars:
            x = round(b.get_x() + b.get_width() / 2.0, 6)
            got[x] = got.get(x, 0.0) + b.get_height()
        per_scene = payload['models']['cobevt'][pl.KEY_PER_SCENE]
        heights = []
        for ti, tier in enumerate(('mild', 'moderate', 'severe')):
            for ii, iou in enumerate(('0.5', '0.7')):
                x = round(ti + (ii - 0.5) * 0.34, 6)
                assert abs(got[x] - 1.0) < 1e-12
                heights.append(x)
        # first-segment heights are the 'established beneficial' fractions
        first = {round(b.get_x() + b.get_width() / 2.0, 6): b.get_height()
                 for b in reversed(bars) if b.get_y() == 0.0}
        for ti, tier in enumerate(('mild', 'moderate', 'severe')):
            for ii, iou in enumerate(('0.5', '0.7')):
                cell = per_scene[iou]['pose_error_%s' % tier]
                ben = sum(1 for v in cell.values()
                          if v['verdict'] == 'established beneficial')
                x = round(ti + (ii - 0.5) * 0.34, 6)
                if ben:
                    assert first[x] == ben / len(cell)
        plt.close(fig)


def test_agent_drop_figure_plots_payload_means():
    payload = pl.load_agent_drop(PAYLOADS)
    with style.figure_style():
        fig = agent_drop_figures.build_fixed_vs_union(payload)
        lines = _lines(fig.axes[0])
        assert len(lines) == 6
        k = 0
        for model in ('cobevt', 'v2xvit', 'where2comm'):
            for key in ('ap_fixed', 'ap_union'):
                t = payload['models'][model]['tiers']
                want = [t[x]['0.7'][key]['mean']
                        for x in ('none', 'mild', 'moderate', 'severe')]
                assert list(lines[k].get_ydata()) == want
                k += 1
        plt.close(fig)


def _grid_float_rows():
    return pl.load_grid_rows_float(PAYLOADS)


def _raw_grid():
    with open(os.path.join(PAYLOADS, 'fault_inference_grid.csv'),
              encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def test_agent_drop_by_stratum_bars_are_grid_values():
    raw = _raw_grid()
    with style.figure_style(style.GRID_FAMILY_RC):
        fig = grid_figures.build_agent_drop_by_stratum(_grid_float_rows())
        top = fig.axes[:3]
        for ax, model in zip(top, ('cobevt', 'v2xvit', 'where2comm')):
            heights = [p.get_height() for p in ax.patches]
            want = []
            for n in ('2', '3', '4', '5'):
                for tier in ('mild', 'moderate', 'severe'):
                    hit = [r for r in raw if r['model'] == model
                           and r['injector'] == 'agent_drop'
                           and r['severity_tier'] == tier
                           and pl.parse_extra(r['extra_metric'])
                           ['scene_agent_count'] == n]
                    assert len(hit) == 1
                    want.append(float(hit[0]['faulty_ap70']))
            assert heights == want
        plt.close(fig)


def test_severe_tier_ordering_bars_are_grid_values():
    raw = _raw_grid()
    with style.figure_style(style.GRID_FAMILY_RC):
        fig = grid_figures.build_severe_tier_ordering(_grid_float_rows())
        widths = sorted(round(p.get_width(), 9) for p in fig.axes[0].patches
                        if isinstance(p, Rectangle)
                        and abs(p.get_height() - 0.19) < 1e-9)
        plt.close(fig)
    want = []
    for r in raw:
        if r['severity_tier'] != 'severe' or r['injector'] == 'none':
            continue
        if r['injector'] == 'agent_drop' and \
                pl.parse_extra(r['extra_metric'])['scene_agent_count'] != '2':
            continue
        want.append(round(float(r['rel_drop_ap70']), 9))
    assert len(want) == 4 * 5 + 3 * 2
    assert widths == sorted(want)


def test_degradation_all_models_lines_are_grid_values():
    raw = _raw_grid()
    with style.figure_style(style.GRID_FAMILY_RC):
        fig = grid_figures.build_degradation_all_models(_grid_float_rows())
        for ax, model in zip(fig.axes, grid_figures.MODELS):
            lines = _lines(ax)
            clean = [r for r in raw if r['model'] == model
                     and r['injector'] == 'none'][0]
            assert list(lines[0].get_ydata()) == [float(clean['clean_ap70'])] * 2
            injs = ['lidar_snow', 'lidar_fog', 'points_reduce', 'pose_error',
                    'latency', 'missing_modality']
            if model == 'cosdh':
                injs = injs[:-1]
            assert len(lines) == 1 + len(injs)
            for ln, inj in zip(lines[1:], injs):
                want = [float([r for r in raw if r['model'] == model
                               and r['injector'] == inj
                               and r['severity_tier'] == t][0]['faulty_ap70'])
                        for t in ('mild', 'moderate', 'severe')]
                assert list(ln.get_ydata()) == want
        plt.close(fig)


def test_figure_negative_control(tmp_path):
    d = os.path.join(str(tmp_path), 'payloads')
    shutil.copytree(PAYLOADS, d)
    p = os.path.join(d, 'agent_drop_fixed_vs_union.json')
    with open(p, encoding='utf-8') as fh:
        obj = json.load(fh)
    obj['models']['cobevt']['tiers']['severe']['0.7']['ap_union']['mean'] \
        += 0.01
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1)
    out = os.path.join(str(tmp_path), 'figs')
    agent_drop_figures.render(d, out)
    same = os.path.join(str(tmp_path), 'figs_same')
    agent_drop_figures.render(PAYLOADS, same)            # unperturbed, same machine
    with open(os.path.join(out, agent_drop_figures.FIGURE_FILE), 'rb') as a, \
            open(os.path.join(same, agent_drop_figures.FIGURE_FILE), 'rb') as b:
        assert pdf_content.content(a.read()) != pdf_content.content(b.read())
    with style.figure_style():
        fig = agent_drop_figures.build_fixed_vs_union(pl.load_agent_drop(d))
        y = list(_lines(fig.axes[0])[1].get_ydata())
        plt.close(fig)
    assert y[-1] == obj['models']['cobevt']['tiers']['severe']['0.7'][
        'ap_union']['mean']
