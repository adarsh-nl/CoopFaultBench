# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""The licence boundary: CC BY-NC 4.0 code stays in injectors/lidar_weather/, Apache-2.0 code stays in
injectors/_vendor/imagecorruptions/, and every source file carries an SPDX header."""
import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, 'coopfaultbench')
NC_DIR = os.path.join(PKG, 'injectors', 'lidar_weather')
APACHE_DIR = os.path.join(PKG, 'injectors', '_vendor', 'imagecorruptions')
MC_DIR = os.path.join(PKG, 'injectors', '_vendor', 'multicorrupt')


def py_files(*roots):
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in ('__pycache__', '.git')]
            for f in sorted(fns):
                if f.endswith('.py'):
                    yield os.path.join(dp, f)


def spdx(path):
    with open(path, encoding='utf-8') as fh:
        head = fh.read(400)
    m = re.search(r'SPDX-License-Identifier:\s*([^\n]+)', head)
    return m.group(1).strip() if m else None


def test_every_python_file_has_spdx():
    files = list(py_files(PKG, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'tests')))
    assert len(files) > 30
    missing = [os.path.relpath(f, ROOT) for f in files if spdx(f) is None]
    assert not missing, missing


def test_licence_of_each_area():
    for f in py_files(PKG):
        lic = spdx(f)
        if f.startswith(NC_DIR + os.sep):
            assert lic == 'CC-BY-NC-4.0', f
        elif f.startswith(APACHE_DIR + os.sep):
            assert lic in ('Apache-2.0', 'Apache-2.0 AND MIT'), f
        else:
            assert lic == 'MIT', (f, lic)
    for d in (NC_DIR, APACHE_DIR, MC_DIR):
        assert os.path.isfile(os.path.join(d, 'LICENSE')) and os.path.isfile(os.path.join(d, 'NOTICE')), d


# Functions of the Hahner et al. fog/snow code (CC BY-NC 4.0). No module outside lidar_weather/ may define them.
NC_NAMES = {'ParameterSet', 'get_integral_dict', 'P_R_fog_hard', 'P_R_fog_soft', 'simulate_fog',
            'compute_occupancy', 'snowfall_rate_to_rainfall_rate', 'estimate_laser_parameters',
            'process_single_channel', 'compute_occlusion_dict',
            'tangent_angles_to_interval_angles', 'do_angles_intersect_particles', 'angles_to_lines',
            'tangent_lines_to_tangent_angles', 'tangents_from_origin', 'distances_of_points_to_lines',
            'get_occlusions', 'received_power', 'xsi', 'simulate_snow_griffin', 'sekhon_srivastava',
            'gunn_marshall', 'dart_throwing'}


def defined_names(path):
    tree = ast.parse(open(path, encoding='utf-8').read())
    return {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_nc_code_defined_only_inside_its_directory():
    inside = set()
    for f in py_files(NC_DIR):
        inside |= defined_names(f)
    assert NC_NAMES <= inside, NC_NAMES - inside          # the check can fail: the names exist there
    for f in py_files(PKG, os.path.join(ROOT, 'scripts')):
        if f.startswith(NC_DIR + os.sep):
            continue
        clash = defined_names(f) & NC_NAMES
        assert not clash, (f, clash)


# Two helpers of unknown licence in the snow code as received via MultiCorrupt were replaced by MIT clean-room code
# (injectors/_snow_helpers.py).
REMOVED_NAMES = {'ransac_polyfit', 'binary_angle_search'}


def test_unknown_licence_helpers_are_gone():
    found = [(os.path.relpath(f, ROOT), n) for f in py_files(PKG, os.path.join(ROOT, 'scripts'))
             for n in defined_names(f) & REMOVED_NAMES]
    assert not found, found
    snow = open(os.path.join(NC_DIR, 'snow_simulation.py'), encoding='utf-8').read()
    assert 'ransac_polynomial_fit(' in snow and 'find_sorted_index(' in snow   # the check can fail


def _lines(path):
    out = set()
    for ln in open(path, encoding='utf-8'):
        s = re.sub(r'\s+', '', ln)
        if len(s) >= 30 and not s.startswith('#'):
            out.add(s)
    return out


def test_no_copied_lines_across_the_boundary():
    """No non-trivial line of the CC BY-NC or Apache files appears in any MIT file."""
    protected = set()
    for d in (NC_DIR, APACHE_DIR):
        for f in py_files(d):
            protected |= _lines(f)
    assert len(protected) > 200
    for f in py_files(PKG, os.path.join(ROOT, 'scripts')):
        if f.startswith(NC_DIR + os.sep) or f.startswith(APACHE_DIR + os.sep):
            continue
        shared = _lines(f) & protected
        assert not shared, (os.path.relpath(f, ROOT), sorted(shared)[:3])
