# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Table rebuild gate.

Regenerates every table from ``results/payloads`` into a temporary directory
and requires byte equality with the shipped ``results/tables/*``. The shipped
canonical tables are the camera-ready files, with internal codenames in a
label, headings and generator comments renamed to plain names (see
``results/MANIFEST.md``); the others are generator outputs.

A gate that cannot fail must not pass, so this module also
* asserts the exact set (and count) of compared files,
* runs one negative control per input source (fault-sweep payload,
  agent-drop payload, grid CSV, grid cells, grid model extract): a perturbed
  copy of the payload must change the regenerated table,
* re-derives printed numbers directly from payload fields.

ONE KNOWN, DOCUMENTED MISMATCH: ``model_reproduction.tex``. The camera-ready
table prints ``n/a`` for Where2comm's published AP, while the grid manifest
in the provenance tag records 0.82 / 0.534 for it, so the regenerated table
prints ``0.8200 & 0.5340``. The numbers are not tweaked; the gate asserts
that this is the ONLY difference, line for line.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from coopfaultbench.reporting import payloads as pl  # noqa: E402
from coopfaultbench.reporting.tables import (build_all_tables,  # noqa: E402
                                             write_tables)

PAYLOADS = os.path.join(REPO, 'results', 'payloads')
TABLES = os.path.join(REPO, 'results', 'tables')

EXPECTED_TABLES = sorted([
    'fault_sweep_cobevt.tex', 'fault_sweep_v2xvit.tex',
    'fault_sweep_where2comm.tex', 'fault_sweep_tables.md',
    'fault_sweep_cobevt.csv', 'fault_sweep_v2xvit.csv',
    'fault_sweep_where2comm.csv',
    'agent_drop_fixed_vs_union.tex', 'agent_drop_fixed_vs_union.md',
    'agent_drop_fixed_vs_union.csv',
    'model_reproduction.tex', 'severe_tier_ordering.tex',
    'agent_drop_by_stratum.tex',
])

# file -> [(line in shipped canonical file, line regenerated from payloads)]
KNOWN_MISMATCH = {
    'model_reproduction.tex': [(
        r'Where2comm & V2XSET & 2834 & True & True & 0.2/0.2 & 0.8197 & '
        r'0.5340 & n/a & n/a & 31/26 & yes \\',
        r'Where2comm & V2XSET & 2834 & True & True & 0.2/0.2 & 0.8197 & '
        r'0.5340 & 0.8200 & 0.5340 & 31/26 & yes \\',
    )],
}


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def _line_diff(a, b):
    la = a.decode('utf-8').split('\n')
    lb = b.decode('utf-8').split('\n')
    assert len(la) == len(lb), 'line count differs'
    return [(x, y) for x, y in zip(la, lb) if x != y]


@pytest.fixture(scope='module')
def regenerated(tmp_path_factory):
    """Run the public entry point, scripts/make_tables.py, into a tmp dir."""
    out = str(tmp_path_factory.mktemp('tables'))
    res = subprocess.run([sys.executable, '-B',
                          os.path.join(REPO, 'scripts', 'make_tables.py'),
                          '--payloads', PAYLOADS, '--out', out],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert res.returncode == 0, res.stderr.decode()
    return out


def test_script_and_library_agree(regenerated, tmp_path):
    lib = str(tmp_path / 'lib')
    write_tables(PAYLOADS, lib)
    for name in EXPECTED_TABLES:
        assert _read(os.path.join(lib, name)) == \
            _read(os.path.join(regenerated, name)), name


def test_table_set_is_exact(regenerated):
    shipped = sorted(os.listdir(TABLES))
    assert shipped == EXPECTED_TABLES
    assert sorted(os.listdir(regenerated)) == EXPECTED_TABLES
    assert sorted(build_all_tables(PAYLOADS)) == EXPECTED_TABLES


def test_regenerated_tables_match_shipped(regenerated):
    compared = 0
    identical = 0
    for name in EXPECTED_TABLES:
        a = _read(os.path.join(TABLES, name))
        b = _read(os.path.join(regenerated, name))
        assert len(a) > 0 and len(b) > 0, name
        compared += 1
        if name in KNOWN_MISMATCH:
            assert a != b, '%s now reproduces; drop it from KNOWN_MISMATCH' \
                % name
            assert _line_diff(a, b) == KNOWN_MISMATCH[name], name
        else:
            assert a == b, '%s differs from its regeneration' % name
            identical += 1
    assert compared == len(EXPECTED_TABLES) == 13
    assert identical == 12


def test_regeneration_is_deterministic(regenerated):
    again = build_all_tables(PAYLOADS)
    for name, data in again.items():
        assert data == _read(os.path.join(regenerated, name)), name


# ------------------------------------------------------- negative controls
def _copy_payloads(tmp_path):
    dst = os.path.join(str(tmp_path), 'payloads')
    shutil.copytree(PAYLOADS, dst)
    return dst


def _rewrite_json(path, mutate):
    with open(path, 'r', encoding='utf-8') as fh:
        obj = json.load(fh)
    mutate(obj)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1)


def _changed(payload_dir):
    """Tables whose regeneration from ``payload_dir`` differs from the
    regeneration from the shipped payloads."""
    base = build_all_tables(PAYLOADS)
    new = build_all_tables(payload_dir)
    assert sorted(new) == sorted(base)
    return sorted(k for k in new if new[k] != base[k])


def test_negative_control_fault_sweep_payload(tmp_path):
    d = _copy_payloads(tmp_path)

    def mutate(o):
        o['models']['cobevt'][pl.KEY_PER_CONDITION]['0.7']['clean'][
            'ap_cp_ci'][0] += 0.01
    _rewrite_json(os.path.join(d, 'fault_sweep_fixed_reference.json'), mutate)
    assert _changed(d) == ['fault_sweep_cobevt.csv', 'fault_sweep_cobevt.tex',
                           'fault_sweep_tables.md']


def test_negative_control_agent_drop_payload(tmp_path):
    d = _copy_payloads(tmp_path)

    def mutate(o):
        o['models']['v2xvit']['tiers']['severe']['0.7']['ap_fixed']['mean'] \
            += 0.01
    _rewrite_json(os.path.join(d, 'agent_drop_fixed_vs_union.json'), mutate)
    assert _changed(d) == ['agent_drop_fixed_vs_union.csv',
                           'agent_drop_fixed_vs_union.md',
                           'agent_drop_fixed_vs_union.tex']


def test_negative_control_grid_csv(tmp_path):
    d = _copy_payloads(tmp_path)
    p = os.path.join(d, 'fault_inference_grid.csv')
    with open(p, 'r', encoding='utf-8', newline='') as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    hits = [r for r in rows[1:] if r[hdr.index('model')] == 'cosdh'
            and r[hdr.index('injector')] == 'lidar_fog'
            and r[hdr.index('severity_tier')] == 'severe']
    assert len(hits) == 1
    k = hdr.index('rel_drop_ap70')
    hits[0][k] = '%.6f' % (float(hits[0][k]) + 0.01)
    with open(p, 'w', encoding='utf-8', newline='') as fh:
        csv.writer(fh).writerows(rows)
    assert _changed(d) == ['severe_tier_ordering.tex']


def test_negative_control_grid_cells_and_models(tmp_path):
    d = _copy_payloads(tmp_path)

    def flip(o):
        o['wild_setting_shipped']['xyz_std'] = 0.3
    _rewrite_json(os.path.join(d, 'cells', 'cobevt', 'none', 'clean.json'),
                  flip)
    assert _changed(d) == ['model_reproduction.tex']

    d2 = _copy_payloads(tmp_path / 'second')

    def pub(o):
        o['models']['cobevt']['published']['ap50'] = 0.85
    _rewrite_json(os.path.join(d2, 'fault_inference_models.json'), pub)
    assert _changed(d2) == ['model_reproduction.tex']


# --------------------------------------------- numbers trace to payloads
def _tex_body_rows(text):
    lines = text.split('\n')
    start = lines.index(r'\midrule') + 1
    end = lines.index(r'\bottomrule')
    return [ln for ln in lines[start:end] if ln != r'\midrule']


def test_fault_sweep_tex_numbers_trace_to_payload():
    payload = pl.load_fault_sweep(PAYLOADS)
    n_checked = 0
    for model in pl.MODELS:
        with open(os.path.join(TABLES, 'fault_sweep_%s.tex' % model),
                  encoding='utf-8') as fh:
            body = _tex_body_rows(fh.read())
        assert len(body) == len(pl.CONDITIONS)
        for line, cond in zip(body, pl.CONDITIONS):
            cells = line[:-len(r' \\')].split(' & ')
            assert cells[0] == pl.cond_label(cond)
            for j, iou in enumerate(pl.IOUS):
                d = payload['models'][model][pl.KEY_PER_CONDITION][iou][cond]
                api, apc, gain = cells[1 + 3 * j: 4 + 3 * j]
                assert api == '%.4f' % d['ap_indiv_ci'][0]
                assert apc == '%.4f' % d['ap_cp_ci'][0]
                g = d['gain_ci']
                assert gain.startswith('%+.4f' % g[0])
                assert '[%+.4f, %+.4f]' % (g[1], g[2]) in gain
                star = gain[len('%+.4f' % g[0]):].split(' [')[0]
                want = ('**' if d['holm_reject'] else
                        '*' if d['uncorrected_reject'] else '')
                assert star == want
                n_checked += 1
    assert n_checked == 3 * 16 * 2


def test_fault_sweep_csv_values_are_payload_values():
    payload = pl.load_fault_sweep(PAYLOADS)
    for model in pl.MODELS:
        with open(os.path.join(TABLES, 'fault_sweep_%s.csv' % model),
                  encoding='utf-8', newline='') as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 32
        for r in rows:
            d = payload['models'][model][pl.KEY_PER_CONDITION][r['iou']][
                r['condition']]
            assert float(r['ap_individual']) == d['ap_indiv_ci'][0]
            assert float(r['ap_cp']) == d['ap_cp_ci'][0]
            assert [float(r['gain_mean']), float(r['gain_ci_lo']),
                    float(r['gain_ci_hi'])] == d['gain_ci']
            assert r['holm_significant'] == str(d['holm_reject'])


def test_agent_drop_table_values_trace_to_payload():
    payload = pl.load_agent_drop(PAYLOADS)
    tier_of = {'%.2f' % v: k for k, v in payload['tiers'].items()}
    with open(os.path.join(TABLES, 'agent_drop_fixed_vs_union.csv'),
              encoding='utf-8', newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 18
    seen = set()
    for r in rows:
        tier = tier_of[r['p_drop']]
        assert tier != 'none'
        c = payload['models'][r['model']]['tiers'][tier][r['iou']]
        assert int(r['n_draws']) == payload['M'] == 100
        want = {
            'ap_union_mean': c['ap_union']['mean'],
            'ap_union_mc_range_lo': c['ap_union']['ci'][0],
            'ap_union_mc_range_hi': c['ap_union']['ci'][1],
            'ap_fixed_mean': c['ap_fixed']['mean'],
            'ap_fixed_mc_range_lo': c['ap_fixed']['ci'][0],
            'ap_fixed_mc_range_hi': c['ap_fixed']['ci'][1],
            'dap_mean': c['mean'],
            'dap_mc_range_lo': c['ci'][0],
            'dap_mc_range_hi': c['ci'][1],
        }
        for k, v in want.items():
            assert r[k] == '%.4f' % v, (r['model'], tier, r['iou'], k)
        seen.add((r['model'], tier, r['iou']))
    assert len(seen) == 18
    # the tex and md carry the same 4-dp strings
    with open(os.path.join(TABLES, 'agent_drop_fixed_vs_union.tex'),
              encoding='utf-8') as fh:
        tex = fh.read()
    assert 'Monte Carlo agent-draw range (not a confidence interval)' in tex
    assert len(_tex_body_rows(tex)) == 18
    for r in rows:
        assert '%s [%s, %s]' % (r['ap_fixed_mean'], r['ap_fixed_mc_range_lo'],
                                r['ap_fixed_mc_range_hi']) in tex


def _grid_rows():
    with open(os.path.join(PAYLOADS, 'fault_inference_grid.csv'),
              encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def test_grid_tables_trace_to_grid_csv():
    rows = _grid_rows()
    assert len(rows) == 109
    assert 'job_id' not in rows[0]

    def stratum(r):
        return pl.parse_extra(r['extra_metric']).get('scene_agent_count')

    # severe-tier ordering: every printed value is a severe-tier rel_drop_ap70
    with open(os.path.join(TABLES, 'severe_tier_ordering.tex'),
              encoding='utf-8') as fh:
        body = _tex_body_rows(fh.read())
    assert len(body) == 7
    models = ['cobevt', 'v2xvit', 'where2comm', 'cosdh']
    label_to_inj = {'LiDAR snow': 'lidar_snow', 'LiDAR fog': 'lidar_fog',
                    'Points reduce': 'points_reduce',
                    'Pose error': 'pose_error', 'Comm. latency': 'latency',
                    'Missing LiDAR': 'missing_modality',
                    'Agent drop*': 'agent_drop'}
    for line in body:
        cells = line[:-len(r' \\')].split(' & ')
        inj = label_to_inj[cells[1]]
        for m, cell in zip(models, cells[2:6]):
            hit = [r for r in rows if r['model'] == m
                   and r['injector'] == inj and r['severity_tier'] == 'severe'
                   and (inj != 'agent_drop' or stratum(r) == '2')]
            if m == 'cosdh' and inj in ('agent_drop', 'missing_modality'):
                assert hit == [] and cell == 'N/A'
            else:
                assert len(hit) == 1
                assert cell == '%.3f' % float(hit[0]['rel_drop_ap70'])

    # agent drop by stratum: union-reference faulty_ap70 per stratum
    with open(os.path.join(TABLES, 'agent_drop_by_stratum.tex'),
              encoding='utf-8') as fh:
        body = _tex_body_rows(fh.read())
    assert len(body) == 12
    lab = {'CoBEVT': 'cobevt', 'V2X-ViT': 'v2xvit', 'Where2comm': 'where2comm'}
    for line in body:
        cells = line[:-len(r' \\')].split(' & ')
        m, s = lab[cells[0]], cells[1]
        for tier, cell in zip(('mild', 'moderate', 'severe'), cells[5:8]):
            hit = [r for r in rows if r['model'] == m
                   and r['injector'] == 'agent_drop'
                   and r['severity_tier'] == tier and stratum(r) == s]
            assert len(hit) == 1
            assert cell == '%.4f' % float(hit[0]['faulty_ap70'])
    assert not any(r['model'] == 'cosdh' and r['injector'] in
                   ('agent_drop', 'missing_modality') for r in rows)

    # model/reproduction: clean AP columns are the grid's clean rows
    with open(os.path.join(TABLES, 'model_reproduction.tex'),
              encoding='utf-8') as fh:
        body = _tex_body_rows(fh.read())
    lab['CoSDH'] = 'cosdh'
    for line in body:
        cells = line[:-len(r' \\')].split(' & ')
        m = lab[cells[0]]
        hit = [r for r in rows if r['model'] == m and r['injector'] == 'none'
               and r['severity_tier'] == 'clean']
        assert len(hit) == 1
        assert cells[2] == hit[0]['n_frames']
        assert cells[6] == '%.4f' % float(hit[0]['clean_ap50'])
        assert cells[7] == '%.4f' % float(hit[0]['clean_ap70'])
        assert cells[10] == '%d/26' % sum(1 for r in rows if r['model'] == m)

    # model/reproduction: remaining columns from the model extract and the
    # clean cell; Where2comm's Pub AP is the documented known mismatch.
    models = pl.load_grid_models(PAYLOADS)['models']
    for line in body:
        cells = line[:-len(r' \\')].split(' & ')
        m = lab[cells[0]]
        mo = models[m]
        assert cells[1] == mo['dataset'].upper()
        ws = (pl.load_grid_cell(PAYLOADS, m, 'none', 'clean')
              .get('wild_setting_shipped') or {})
        if ws:
            assert cells[3:6] == [str(ws['loc_err']), str(ws['async']),
                                  '%s/%s' % (ws['xyz_std'], ws['ryp_std'])]
        else:
            assert m == 'cosdh' and cells[3:6] == ['n/a', 'n/a', 'n/a']
        pub = mo['published']
        if m == 'where2comm':
            assert cells[8:10] == ['n/a', 'n/a']          # known mismatch
            assert ('%.4f' % pub['ap50'], '%.4f' % pub['ap70']) == \
                ('0.8200', '0.5340')
        elif pub:
            assert cells[8:10] == ['%.4f' % pub['ap50'], '%.4f' % pub['ap70']]
        else:
            assert cells[8:10] == ['n/a', 'n/a']
        assert cells[11] == ('yes' if (mo['graded'] or m == 'where2comm')
                             else 'no')
