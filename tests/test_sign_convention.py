# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""One sign convention everywhere: dAP = AP_fixed - AP_union (the paper's definition).

dAP is negative when the union reference inflates AP. These tests fail if a payload, a shipped table or a shipped
document prints a known agent-drop cell with the opposite sign, or defines dAP the other way round."""
import csv
import glob
import json
import os
import re

from coopfaultbench.evaluation import dual_ap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, 'results')
PAYLOAD = os.path.join(RES, 'payloads', 'agent_drop_fixed_vs_union.json')
BOOT = os.path.join(RES, 'payloads', 'agent_drop_scene_bootstrap.json')
NAME = {'cobevt': 'CoBEVT', 'v2xvit': 'V2X-ViT', 'where2comm': 'Where2comm'}
P_DROP = {'mild': '0.25', 'moderate': '0.50', 'severe': '0.75'}


def _cells():
    p = json.load(open(PAYLOAD))
    return {(m, t, iou): p['models'][m]['tiers'][t][iou]
            for m in NAME for t in P_DROP for iou in ('0.5', '0.7')}


def test_definition_in_code():
    """The scorer computes dAP as AP_fixed - AP_union."""
    import inspect
    src = inspect.getsource(dual_ap)
    assert "'dap': ap_f - ap_u" in src.replace('"', "'")


def test_payload_field_is_fixed_minus_union():
    cells = _cells()
    assert len(cells) == 18
    for k, c in cells.items():
        assert abs(c['mean'] - (c['ap_fixed']['mean'] - c['ap_union']['mean'])) < 1e-12, k
        assert c['mean'] < 0, k                       # every published cell: the union reference inflates AP


def test_bootstrap_payload_same_sign():
    if not os.path.isfile(BOOT):
        return
    b = json.load(open(BOOT))['cells']
    for (m, t, iou), c in _cells().items():
        e = b['%s|%s|%s' % (m, t, iou)]
        assert abs(e['plugin_mc_mean'] - c['mean']) < 1e-12, (m, t, iou)
        assert abs(e['plugin_ap_fixed'] - e['plugin_ap_union'] - e['plugin_mc_mean']) < 1e-12, (m, t, iou)
        lo, hi = e['primary']['ci']
        assert lo <= hi and lo < 0, (m, t, iou)       # an interval of dAP, not of its negation


def _table_rows():
    """(cell, printed dAP string) from every shipped rendering of the agent-drop table."""
    out = []
    with open(os.path.join(RES, 'tables', 'agent_drop_fixed_vs_union.csv')) as f:
        for r in csv.DictReader(f):
            t = {v: k for k, v in P_DROP.items()}[r['p_drop'] if len(r['p_drop']) == 4 else r['p_drop'] + '0']
            out.append(((r['model'], t, r['iou']), r['dap_mean'], 'csv'))
    inv = {v: k for k, v in NAME.items()}
    for ext, sep in (('md', '|'), ('tex', '&')):
        for line in open(os.path.join(RES, 'tables', 'agent_drop_fixed_vs_union.' + ext), encoding='utf-8'):
            cols = [c.strip() for c in line.strip().strip('|').replace('\\\\', '').split(sep)]
            if len(cols) == 6 and cols[0] in inv:
                t = {v: k for k, v in P_DROP.items()}[cols[1]]
                out.append(((inv[cols[0]], t, cols[2]), cols[5].split()[0], ext))
    return out


def test_shipped_agent_drop_tables_print_signed_dap():
    cells = _cells()
    rows = _table_rows()
    assert len(rows) == 3 * 18, len(rows)
    for cell, printed, where in rows:
        assert printed == '%.4f' % cells[cell]['mean'], (where, cell, printed)


# Definitions the other way round, in any shipped text.
REVERSED = [r'dAP\s*=\s*AP_?\{?union', r'(?:d|\\Delta\s*\$?\s*)AP\$?\s*\$?=\s*\$?\s*AP\$?_\{?\\mathrm\{union',
            r'AP_union\s*[-−]\s*AP_fixed', r'\bunion\s*[-−]\s*fixed\b']
DAP_LINE = re.compile(r'dAP|\\Delta\$?\s*AP|Δ\s*AP')


def _scan(text, magnitudes):
    """Problems in one text: reversed definitions, and a known cell magnitude printed unsigned on a dAP line."""
    bad = [m.group(0) for r in REVERSED for m in re.finditer(r, text)]
    for line in text.splitlines():
        if not DAP_LINE.search(line):
            continue
        for s in magnitudes:
            for m in re.finditer(r'(?<![\d.])' + re.escape(s) + r'(?!\d)', line):
                if line[max(0, m.start() - 1):m.start()] not in ('-', '−'):
                    bad.append(line.strip()[:120])
    return bad


def _shipped_text_files():
    fs = ['README.md', 'CHANGELOG.md', 'ROADMAP.md'] + glob.glob(os.path.join(REPO, 'docs', '*.md'))
    fs += [os.path.join(RES, 'README.md'), os.path.join(RES, 'MANIFEST.md')]
    fs += glob.glob(os.path.join(RES, 'tables', '*'))
    fs += glob.glob(os.path.join(REPO, 'coopfaultbench', '**', '*.py'), recursive=True)
    fs += glob.glob(os.path.join(REPO, 'scripts', '*.py'))
    return [os.path.join(REPO, f) if not os.path.isabs(f) else f for f in fs]


def test_no_shipped_text_prints_the_opposite_sign():
    mags = sorted({'%.4f' % abs(c['mean']) for c in _cells().values()})
    files = _shipped_text_files()
    assert len(files) > 30
    bad = []
    for f in files:
        if f.endswith(os.path.join('tests', 'test_sign_convention.py')):
            continue
        for b in _scan(open(f, encoding='utf-8', errors='replace').read(), mags):
            bad.append((os.path.relpath(f, REPO), b))
    assert not bad, bad[:5]


def test_scanner_can_fail():
    mags = ['0.0196']
    assert _scan('| CoBEVT | 0.75 | 0.7 | ... | dAP 0.0196 |', mags)                 # unsigned on a dAP line
    assert _scan('Delta = AP_union - AP_fixed', mags)                              # reversed definition
    assert _scan(r'$\Delta$AP $=$ AP$_{\mathrm{union}}$ $-$ AP$_{\mathrm{fixed}}$', mags)
    assert not _scan('| CoBEVT | 0.75 | 0.7 | ... | dAP -0.0196 |', mags)
    assert not _scan('dAP = AP_fixed - AP_union', mags)
    assert not _scan('Gain +0.0196 (a fault-sweep line, another quantity)', mags)
