# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Integrity of the released ``results/`` tree and of the reporting code:
checksums, manifest coverage, sanitization invariants, licence headers,
Python 3.7 syntax, and the plain-name rule (no internal codenames or private
source paths in any released file)."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results')
PAYLOADS = os.path.join(RESULTS, 'payloads')

PINNED = {
    'fault_sweep_fixed_reference.json':
        'c89d8b6b50a7177971fa7334d46d253cec079a617a99626049c9cc3aa38e97b6',
    'agent_drop_fixed_vs_union.json':
        '61e59f5181df7938e5884c700e037079f469c37e631b2b5f17ad2737e6ba49cb',
}
# Payloads produced after the provenance tag by CPU jobs (scene-level bootstrap, reference shrinkage).
# A release ships each of them only if its job passed every gate.
POST_TAG = ('payloads/agent_drop_scene_bootstrap.json', 'payloads/reference_shrinkage.json',
            'payloads/agent_drop_scene_bootstrap_replicates.npz')
HEADER = ('# SPDX-License-Identifier: MIT\n'
          '# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.\n')
LICENCE_LINE = ('Results in this directory are licensed under CC BY 4.0 '
                '(https://creativecommons.org/licenses/by/4.0/)')

OWN_PY = (
    ['coopfaultbench/__init__.py']
    + ['coopfaultbench/reporting/%s' % f for f in sorted(os.listdir(
        os.path.join(REPO, 'coopfaultbench', 'reporting')))
       if f.endswith('.py')]
    + ['scripts/%s.py' % n for n in ('sanitize_payloads', 'make_tables',
                                     'make_figures')]
    + ['tests/%s.py' % n for n in ('test_rebuild_tables',
                                   'test_rebuild_figures',
                                   'test_results_integrity')]
)


def sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def walk(root):
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d != '__pycache__')
        for f in sorted(fn):
            out.append(os.path.relpath(os.path.join(dp, f), root)
                       .replace(os.sep, '/'))
    return sorted(out)


# ------------------------------------------------------------ checksums
def test_sha256sums_cover_and_verify_results():
    with open(os.path.join(RESULTS, 'SHA256SUMS'), encoding='utf-8') as fh:
        lines = [ln.rstrip('\n') for ln in fh if ln.strip()]
    listed = {}
    for ln in lines:
        h, path = ln.split('  ', 1)
        assert re.fullmatch(r'[0-9a-f]{64}', h)
        listed[path] = h
    files = [f for f in walk(RESULTS) if f != 'SHA256SUMS']
    assert sorted(listed) == files
    for f in files:
        assert sha(os.path.join(RESULTS, f)) == listed[f], f
    assert len(files) == 125 + sum(f in files for f in POST_TAG)


def _post_tag(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        import pytest
        pytest.skip('%s is not part of this release' % name)
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def test_scene_bootstrap_payload_structure():
    d = _post_tag(POST_TAG[0])
    assert d['gates'] == {'inputs': 'PASS', 'identity': 'PASS', 'coverage': 'PASS', 'null_control': 'PASS'}
    assert d['meta']['B'] == 1000 and d['meta']['seed'] == 20260826
    assert d['meta']['identity_values_compared'] > 0
    assert len(d['cells']) == 18
    for key, c in d['cells'].items():
        assert key == '%s|%s|%s' % (c['model'], c['tier'], c['iou'])
        p = c['primary']
        lo, hi = p['ci']
        assert lo <= hi
        assert p['excludes_zero'] == (hi < 0 or lo > 0), key
        assert p['verdict'] == ('established' if p['excludes_zero'] else 'not established'), key
        assert not p['holm_reject_18'] or p['excludes_zero'], key
        assert 0.0 <= p['p_boot'] <= 1.0, key
        for n, st in c['per_stratum'].items():
            few = st['n_scenes'] < 5
            assert few == (st['ci_status'] == 'not interpretable: too few scenes'), (key, n)
            assert st['sound'] == (not few), (key, n)


def test_scene_bootstrap_replicates_match_intervals():
    d = _post_tag(POST_TAG[0])
    import numpy as np
    path = os.path.join(RESULTS, POST_TAG[2])
    assert os.path.exists(path)
    with np.load(path) as z:
        for k in z.keys():          # binary file: its array names obey the plain-name rule too
            assert not _bad_tokens(k), k
        for key, c in d['cells'].items():
            v = z['%s|primary' % key]
            assert v.shape == (d['meta']['B'],)
            lo, hi = c['primary']['ci']
            assert np.isclose(np.percentile(v, 2.5), lo, rtol=0, atol=1e-12)
            assert np.isclose(np.percentile(v, 97.5), hi, rtol=0, atol=1e-12)


def test_reference_shrinkage_payload_structure():
    d = _post_tag(POST_TAG[1])
    assert d['gates'] == {'inputs': 'PASS', 'identity': 'PASS', 'null_control': 'PASS'}
    assert d['meta']['M'] == 100
    assert sorted(d['models']) == ['cobevt', 'v2xvit', 'where2comm']
    for m, v in d['models'].items():
        assert sorted(v['tiers']) == ['mild', 'moderate', 'severe']
        for tier, s in v['tiers'].items():
            for part in [s['pooled']] + list(s['per_stratum'].values()):
                lo, hi = part['range_2.5_97.5']
                assert lo <= hi <= 0.0 and part['mean'] <= 0.0, (m, tier)   # the union reference only shrinks
        ego = v['ego_only']
        assert ego['pooled'] <= 0.0 and all(x <= 0.0 for x in ego['per_stratum'].values()), m
        pooled = [v['tiers'][t]['pooled']['mean'] for t in ('mild', 'moderate', 'severe')] + [ego['pooled']]
        assert pooled == sorted(pooled, reverse=True), m      # more agents dropped -> more shrinkage


def test_manifest_has_one_correct_row_per_file():
    with open(os.path.join(RESULTS, 'MANIFEST.md'), encoding='utf-8') as fh:
        text = fh.read()
    rows = re.findall(r'^\| `results/([^`]+)` \| `([0-9a-f]{64})` \|', text,
                      re.M)
    paths = [p for p, _ in rows]
    assert len(paths) == len(set(paths))
    files = [f for f in walk(RESULTS)
             if f not in ('SHA256SUMS', 'MANIFEST.md')]
    assert sorted(paths) == files
    for p, h in rows:
        assert sha(os.path.join(RESULTS, p)) == h, p


def test_pinned_payloads():
    for name, want in PINNED.items():
        assert sha(os.path.join(PAYLOADS, name)) == want


# ------------------------------------------------------- sanitization
# Generic private-string patterns, assembled from fragments so this file
# itself contains no such path.
GENERIC_PRIVATE = re.compile('|'.join([
    '/' + 'home' + '/', '/' + 'local' + '/[0-9]', '/' + 'datasets' + '/',
    '~' + '/', r'[A-Za-z0-9._%+-]+' + '@' + r'[A-Za-z0-9-]+\.[A-Za-z]{2,}']))


def test_grid_csv_is_sanitized():
    with open(os.path.join(PAYLOADS, 'fault_inference_grid.csv'),
              encoding='utf-8', newline='') as fh:
        rows = list(csv.reader(fh))
    assert 'job_id' not in rows[0]
    assert len(rows) == 110 and all(len(r) == len(rows[0]) for r in rows)
    models = [r[0] for r in rows[1:]]
    assert {m: models.count(m) for m in set(models)} == {
        'cobevt': 31, 'v2xvit': 31, 'where2comm': 31, 'cosdh': 16}


def test_cells_are_sanitized():
    cells = walk(os.path.join(PAYLOADS, 'cells'))
    per_model = {}
    for c in cells:
        per_model[c.split('/')[0]] = per_model.get(c.split('/')[0], 0) + 1
        with open(os.path.join(PAYLOADS, 'cells', c), encoding='utf-8') as fh:
            obj = json.load(fh)
        for k in ('model_dir', 'opencood_path', 'job_id', 'frame_stratum'):
            assert k not in obj, (c, k)
    assert per_model == {'cobevt': 26, 'v2xvit': 26, 'where2comm': 26,
                         'cosdh': 20}
    # CoSDH agent-drop / missing-modality cells are absent, never filled
    assert not any(c.startswith(('cosdh/agent_drop/',
                                 'cosdh/missing_modality/')) for c in cells)


def test_no_generic_private_strings():
    for f in _own_text_files():
        with open(os.path.join(REPO, f), encoding='utf-8') as fh:
            assert not GENERIC_PRIVATE.search(fh.read()), f


def test_readme_licence_and_caveats():
    with open(os.path.join(RESULTS, 'README.md'), encoding='utf-8') as fh:
        text = fh.read()
    assert LICENCE_LINE in text
    assert ('"These checkpoints have been retrained and the results differ '
            'slightly from those reported in the paper."') in text
    assert 'not confidence intervals' in text
    assert 'bootstrap means' in text


# ---------------------------------------------------------- code rules
def test_spdx_header_on_own_python_files():
    for rel in OWN_PY:
        with open(os.path.join(REPO, rel), encoding='utf-8') as fh:
            assert fh.read().startswith(HEADER), rel


def test_reporting_package_is_python37_syntax():
    d = os.path.join(REPO, 'coopfaultbench', 'reporting')
    n = 0
    for f in sorted(os.listdir(d)):
        if f.endswith('.py'):
            with open(os.path.join(d, f), encoding='utf-8') as fh:
                src = fh.read()
            if sys.version_info >= (3, 8):
                ast.parse(src, filename=f, feature_version=(3, 7))
            else:                   # on Python 3.7 its own grammar is the 3.7 check
                ast.parse(src, filename=f)
            n += 1
    assert n >= 9


# Plain-name rule. Internal codenames must not appear in released names or
# text. The lists are assembled from fragments so that this file passes the
# same scan. A token is a maximal [a-z0-9] run (case-insensitive); it is
# rejected when it starts with a FORBIDDEN_PREFIX or equals a FORBIDDEN_EXACT
# version-suffix token. Model/dataset names (V2X-ViT, V2XSet, v2xvit) and
# 64-hex digests are allowed.
FORBIDDEN_PREFIX = tuple(
    [c + str(i) for c, rng in (('e', (1, 2, 3)), ('m', (2, 3, 4, 5)),
                               ('n', range(7)), ('q', (1, 2, 3)))
     for i in rng]
    + ['t' + '9', 'phase' + '0', 'a' + 'pf', 'co' + 'ra', 'gt' + 'fix',
       'opt' + '1', 'opt' + '2', 'p' + 'oc'])
FORBIDDEN_EXACT = ('v' + '2', 'v' + '3')
FORBIDDEN_SUBSTRING = ('tools' + '/sweep', 'number' + '_ledger',
                       '_re' + 'con')
# Two-word internal labels that the token rule cannot see ("<word> <digit>"), matched on the original text:
# the track label only with a capital letter or as an identifier, so that English such as "to track a frame"
# passes.
FORBIDDEN_REGEX = (r'(?i)\bphase[\s_-]*[0-9]\b',
                   r'\b[Tt]' + 'rack' + r'[\s_-]*[AB]\b|\b' + 'track' + r'[_-][ab]\b',
                   r'(?i)\b' + 'gate' + r'[\s_-]*[0-9]\b')
ALLOWED_PREFIX = ('v' + '2x',)


def _bad_tokens(text):
    bad = []
    low = text.lower()
    for tok in re.findall(r'[a-z0-9]+', low):
        if tok.startswith(ALLOWED_PREFIX) or re.fullmatch(r'[0-9a-f]{64}',
                                                          tok):
            continue
        if tok in FORBIDDEN_EXACT or tok.startswith(FORBIDDEN_PREFIX):
            bad.append(tok)
    bad += [s for s in FORBIDDEN_SUBSTRING if s in low]
    bad += [m.group(0).lower() for r in FORBIDDEN_REGEX for m in re.finditer(r, text)]
    return bad


# The one documented exception: the byte-identical fault-sweep payload, whose
# two per-model block keys (and one rule string) are internal names that
# cannot change without breaking byte identity with the frozen payload.
BYTE_IDENTICAL_EXCEPTION = 'results/payloads/fault_sweep_fixed_reference.json'


def _own_text_files():
    out = [p for p in OWN_PY]
    out += ['docs/results_schema.md']
    out += ['results/' + f for f in walk(RESULTS)
            if not f.endswith(('.pdf', '.npz'))
            and 'results/' + f != BYTE_IDENTICAL_EXCEPTION]
    return out


def test_no_codenames_in_released_names_and_text():
    names = walk(RESULTS) + [os.path.basename(p) for p in OWN_PY]
    for n in names:
        assert not _bad_tokens(n), n
    checked = 0
    for rel in _own_text_files():
        with open(os.path.join(REPO, rel), encoding='utf-8') as fh:
            bad = _bad_tokens(fh.read())
        assert not bad, (rel, sorted(set(bad)))
        checked += 1
    assert checked >= 130


def test_byte_identical_payload_exception_is_exactly_the_block_keys():
    with open(os.path.join(REPO, BYTE_IDENTICAL_EXCEPTION),
              encoding='utf-8') as fh:
        bad = set(_bad_tokens(fh.read()))
    assert bad == {'q' + '1', 'q' + '2', 'q' + '3'}


def test_codename_scanner_can_fail():
    assert _bad_tokens('see tab:' + 'e' + '3_cobevt') == ['e' + '3']
    assert _bad_tokens('fig_stratified_' + 'v' + '2.py') == ['v' + '2']
    assert _bad_tokens('tools' + '/sweep/x.py') == ['tools' + '/sweep']
    assert _bad_tokens('V2X-ViT on V2XSet, v2xvit') == []
    assert _bad_tokens('with its ' + 'Phase' + ' 0 sha256') == ['phase' + ' 0']
    assert _bad_tokens('see ' + 'Track' + ' A') == ['track' + ' a']
    assert _bad_tokens('track' + '_b results') == ['track' + '_b']
    assert _bad_tokens('Gate' + ' 2 passed') == ['gate' + ' 2']
    assert _bad_tokens('to track a frame; the wrapped-null gate; phase noise') == []
    assert GENERIC_PRIVATE.search('/' + 'home' + '/someone/x')


def test_manifest_cited_repo_files_match():
    """Every repository file that MANIFEST.md cites with a sha256 (e.g. the sanitizer) is the shipped file."""
    text = open(os.path.join(REPO, 'results', 'MANIFEST.md'), encoding='utf-8').read()
    cited = re.findall(r'`((?:scripts|coopfaultbench|tests)/[^`]+)` \(sha256 `([0-9a-f]{64})`\)', text)
    assert len(cited) >= 100                  # the sanitizer is cited once per sanitized cell file
    for path, h in cited:
        with open(os.path.join(REPO, path), 'rb') as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == h, path
