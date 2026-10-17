# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""The README quickstart runs as written and prints what the README says."""
import contextlib
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_quickstart_runs():
    text = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()
    code = re.search(r'## Five-minute quickstart.*?```python\n(.*?)```', text, re.S).group(1)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, 'README-quickstart', 'exec'), {})
    assert "'ap_union': 1.0, 'ap_fixed': 0.75, 'dap': -0.25" in out.getvalue()
    assert 'pose_error' in out.getvalue()
