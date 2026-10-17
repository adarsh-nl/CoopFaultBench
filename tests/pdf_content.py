# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Compare matplotlib PDFs by content, not by compressed bytes (test helper).

The compressed bytes of a PDF's image streams depend on the zlib build that Pillow ships, which differs between CPU
architectures (e.g. x86 vs Apple Silicon), so byte equality of the files is not portable. ``content`` returns what a
reader sees: every object in file order, with each Flate stream decompressed and PNG-predicted image rows decoded to
their pixel bytes. Stream lengths, which follow from the compression, are ignored (inline ``/Length`` values and the
separate length objects matplotlib writes), as are the cross-reference offsets. Everything else -- object numbers,
dictionaries, decompressed content, pixels, the trailer dictionary -- must be equal.
"""
import re
import zlib

_OBJ = re.compile(rb'(\d+) 0 obj\s*(.*?)endobj', re.S)
_STREAM = re.compile(rb'>>\s*stream\r?\n')
_LENGTH = re.compile(rb'/Length \d+(?: 0 R)?')


def _objects(data):
    """[(number, head, payload, (start, end) of payload in data)]; head None for non-stream objects."""
    out = []
    for m in _OBJ.finditer(data):
        body, off = m.group(2), m.start(2)
        s = _STREAM.search(body)
        if s is None:
            out.append((int(m.group(1)), None, body.strip(), None))
            continue
        a, b = s.end(), body.rfind(b'endstream')
        raw = body[a:b]
        if raw.endswith(b'\r\n'):
            raw = raw[:-2]
        elif raw.endswith(b'\n'):
            raw = raw[:-1]
        out.append((int(m.group(1)), body[:s.end()], raw, (off + a, off + a + len(raw))))
    return out


def _param(head, key):
    m = re.search(rb'/' + key + rb' (\d+)', head)
    return int(m.group(1)) if m else None


def _unpredict(data, head):
    """Undo PNG predictors (/Predictor 10-15): return the pixel rows without the per-row filter bytes."""
    colors = _param(head, b'Colors') or 1
    bpc = _param(head, b'BitsPerComponent') or 8
    columns = _param(head, b'Columns') or 1
    rowlen = (colors * bpc * columns + 7) // 8
    bpp = max(1, (colors * bpc + 7) // 8)
    out, prev, i = bytearray(), bytearray(rowlen), 0
    while i < len(data):
        ftype, row = data[i], bytearray(data[i + 1:i + 1 + rowlen])
        i += 1 + rowlen
        for x in range(len(row)):
            left = row[x - bpp] if x >= bpp else 0
            up, ul = prev[x], (prev[x - bpp] if x >= bpp else 0)
            if ftype == 1:
                row[x] = (row[x] + left) & 255
            elif ftype == 2:
                row[x] = (row[x] + up) & 255
            elif ftype == 3:
                row[x] = (row[x] + ((left + up) >> 1)) & 255
            elif ftype == 4:
                p = left + up - ul
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - ul)
                row[x] = (row[x] + (left if pa <= pb and pa <= pc else up if pb <= pc else ul)) & 255
            elif ftype != 0:
                raise ValueError('unknown PNG filter type %d' % ftype)
        out += row
        prev = row
    return bytes(out)


def _decode(head, raw):
    if b'/FlateDecode' not in head:
        return raw
    data = zlib.decompressobj().decompress(raw)
    pred = _param(head, b'Predictor')
    return _unpredict(data, head) if pred and pred >= 10 else data


def content(data):
    """Everything a reader sees, independent of how the streams were compressed."""
    assert data.startswith(b'%PDF-')
    length_objs = {int(x) for x in re.findall(rb'/Length (\d+) 0 R', data)}
    out = []
    for num, head, payload, _ in _objects(data):
        if head is None:
            out.append((num, b'<length>' if num in length_objs else payload))
        else:
            out.append((num, _LENGTH.sub(b'/Length *', head), _decode(head, payload)))
    trailer = data[data.rfind(b'trailer'):data.rfind(b'startxref')]
    return out, trailer


def rewrite_streams(data, edit=None, level=9):
    """Test tool: re-encode every Flate stream (zlib ``level``) after ``edit(number, head, decompressed) ->
    decompressed`` (None: unchanged). Inline /Length values are updated; length objects and xref offsets are
    left stale (``content`` ignores both)."""
    pieces, pos = [], 0
    for num, head, payload, span in _objects(data):
        if head is None or b'/FlateDecode' not in head:
            continue
        dec = zlib.decompressobj().decompress(payload)
        if edit is not None:
            dec = edit(num, head, dec)
        new = zlib.compress(dec, level)
        hs = data.rfind(head, 0, span[0])
        m = re.search(rb'/Length (\d+)( \d+ R)?', head)
        newhead = head
        if m and not m.group(2):                          # a direct length, not a reference
            newhead = head[:m.start()] + b'/Length %d' % len(new) + head[m.end():]
        pieces += [data[pos:hs], newhead, new]
        pos = span[1]
    pieces.append(data[pos:])
    return b''.join(pieces)
