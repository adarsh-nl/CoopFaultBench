# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""The exact transformations that turned the authors' private result files
into the sanitized files under ``results/payloads``.

This script knows nothing about where those private files live: every input
and output is an explicit argument. The authors' private build driver called
these functions once per released file (its sha256 is recorded in
``results/MANIFEST.md``). Every function proves that it changes nothing
except what it says it drops, and is deterministic.

Subcommands::

    copy   IN OUT --sha256 HEX   byte-identical copy, sha256 must match
    grid   IN.csv OUT.csv        drop the ``job_id`` column; every other field
                                 of every row byte-for-byte unchanged
    cell   IN.json OUT.json      drop keys ``model_dir``, ``opencood_path``,
                                 ``job_id``, ``frame_stratum``; other keys keep
                                 their order and values; json.dump(indent=1)
    models IN.json OUT.json      keep only ``models.<m>.{dataset, setting,
                                 expected_frames, graded, published}`` and
                                 ``not_applicable_cells`` of a grid manifest

``--deny SUBSTRING`` (repeatable) adds site-specific strings (host or account
names) that must not appear in any output; they are supplied at run time and
deliberately not written into this public file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys

CELL_DROP_KEYS = ('model_dir', 'opencood_path', 'job_id', 'frame_stratum')
GRID_DROP_COLUMN = 'job_id'
MODEL_FIELDS = ('dataset', 'setting', 'expected_frames', 'graded', 'published')

# Generic private-string patterns (home directories, per-job scratch paths,
# cluster storage paths, home-relative paths, e-mail addresses), assembled
# from fragments so this file itself contains no such path.
_GENERIC = ['/' + 'home' + '/', '/' + 'local' + '/[0-9]',
            '/' + 'datasets' + '/', '~' + '/',
            r'[A-Za-z0-9._%+-]+' + '@' + r'[A-Za-z0-9-]+\.[A-Za-z]{2,}']


def private_pattern(deny=()):
    parts = list(_GENERIC) + [re.escape(d) for d in deny]
    return re.compile('|'.join(parts), re.IGNORECASE)


def _fail(msg):
    raise SystemExit('FATAL: ' + msg)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def copy_exact(src, dst, sha256):
    with open(src, 'rb') as fh:
        data = fh.read()
    if _sha256(data) != sha256:
        _fail('%s: sha256 %s, expected %s' % (src, _sha256(data), sha256))
    shutil.copyfile(src, dst)
    with open(dst, 'rb') as fh:
        if _sha256(fh.read()) != sha256:
            _fail('copy to %s is not byte-identical' % dst)


def sanitize_grid_bytes(raw, pattern):
    """CSV bytes -> CSV bytes without the ``job_id`` column."""
    text = raw.decode('utf-8')
    rows = list(csv.reader(io.StringIO(text, newline='')))
    # Round-trip proof: csv.writer reproduces the original bytes exactly
    # (CRLF line ends, quoting), so re-writing changes nothing by itself.
    buf = io.StringIO(newline='')
    csv.writer(buf).writerows(rows)
    if buf.getvalue().encode('utf-8') != raw:
        _fail('csv.writer does not round-trip the input byte-for-byte')
    header = rows[0]
    if header.count(GRID_DROP_COLUMN) != 1:
        _fail('expected exactly one %r column' % GRID_DROP_COLUMN)
    if any(len(r) != len(header) for r in rows):
        _fail('ragged CSV row')
    k = header.index(GRID_DROP_COLUMN)
    buf = io.StringIO(newline='')
    csv.writer(buf).writerows([r[:k] + r[k + 1:] for r in rows])
    out = buf.getvalue()
    if pattern.search(out):
        _fail('private string survives in the sanitized grid')
    return out.encode('utf-8')


def _round_trips(raw, obj):
    return any(json.dumps(obj, indent=ind) == raw.rstrip('\n')
               for ind in (1, 2, 4))


def sanitize_cell_text(raw, pattern):
    """Per-cell JSON text -> sanitized JSON text (indent=1)."""
    obj = json.loads(raw)
    # Value-fidelity proof: the parsed object re-serialises to the input
    # text, so dropping keys is the only change made.
    if not _round_trips(raw, obj):
        _fail('cell does not round-trip through json')
    clean = {k: v for k, v in obj.items() if k not in CELL_DROP_KEYS}
    text = json.dumps(clean, indent=1)
    if pattern.search(text):
        _fail('private string survives in the sanitized cell')
    if json.loads(text) != clean:
        _fail('sanitized cell does not re-parse equal')
    return text


def project_models_text(manifest, pattern):
    """Grid manifest object -> JSON text of the released projection."""
    obj = {'models': {op: {f: mo[f] for f in MODEL_FIELDS}
                      for op, mo in manifest['models'].items()},
           'not_applicable_cells': list(manifest['not_applicable_cells'])}
    text = json.dumps(obj, indent=1)
    if pattern.search(text):
        _fail('private string survives in the model projection')
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--deny', action='append', default=[])
    sub = ap.add_subparsers(dest='cmd')
    for name in ('copy', 'grid', 'cell', 'models'):
        p = sub.add_parser(name)
        p.add_argument('src')
        p.add_argument('dst')
        if name == 'copy':
            p.add_argument('--sha256', required=True)
    a = ap.parse_args(argv)
    pattern = private_pattern(a.deny)
    if a.cmd == 'copy':
        copy_exact(a.src, a.dst, a.sha256)
    elif a.cmd == 'grid':
        with open(a.src, 'rb') as fh:
            data = sanitize_grid_bytes(fh.read(), pattern)
        with open(a.dst, 'wb') as fh:
            fh.write(data)
    elif a.cmd == 'cell':
        with open(a.src, encoding='utf-8') as fh:
            text = sanitize_cell_text(fh.read(), pattern)
        with open(a.dst, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(text)
    elif a.cmd == 'models':
        with open(a.src, encoding='utf-8') as fh:
            text = project_models_text(json.load(fh), pattern)
        with open(a.dst, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(text)
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
