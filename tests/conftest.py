# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # tests/golden_cases.py


def _importable(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


HAS_CAMERA = _importable('cv2') and _importable('skimage')
HAS_MPL = _importable('matplotlib')
HAS_OPENCOOD = _importable('opencood.utils.transformation_utils')
HAS_OPENCOOD_EVAL = HAS_OPENCOOD and _importable('opencood.utils.eval_utils')

needs_camera = pytest.mark.skipif(not HAS_CAMERA, reason='camera extra (opencv, scikit-image) not installed')
needs_opencood = pytest.mark.skipif(not HAS_OPENCOOD, reason='OpenCOOD not installed')
needs_opencood_eval = pytest.mark.skipif(not HAS_OPENCOOD_EVAL,
                                         reason='OpenCOOD evaluation (needs torch, shapely) not importable')
