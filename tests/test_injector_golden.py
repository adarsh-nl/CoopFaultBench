# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""The ported injectors reproduce the ORIGINAL code's outputs byte for byte (tests/data/injector_golden.json).

The 30 non-snow cases were recorded with the original code under numpy 1.21.6 (Python 3.7, the environment
of the paper's fault runs) and under numpy 2.5.1; their digests are identical in both, and they are checked on
every numpy version. LiDAR snow output depends on numpy's argpartition tie order (numpy 1.x vs 2.x) and, for
numpy >= 2, on the CPU's SIMD dispatch, so the snow cases are checked only where a golden for this numpy
version (and, for numpy >= 2, this CPU feature set) exists; otherwise they are skipped with the reason.
"""
import json
import os

import numpy as np
import pytest

import golden_cases as gc
from conftest import needs_camera
from coopfaultbench.injectors import (AgentDropInjector, CalibrationErrorInjector, CommLatencyInjector,
                                      LidarFogInjector, LidarSnowInjector, MissingModalityInjector,
                                      OcclusionConfig, PointsReductionInjector, PoseErrorInjector,
                                      SensorOcclusionInjector, TemporalMisalignmentInjector, bernoulli_mask)

GOLD = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'injector_golden.json')))


def _cpu_features():
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats
    except ImportError:
        try:
            from numpy.core._multiarray_umath import __cpu_features__ as feats
        except ImportError:
            return None
    return sorted(k for k, v in feats.items() if v and (k.startswith('AVX512') or k in ('AVX2', 'X86_V3', 'X86_V4')))


def snow_golden():
    """The snow digests for this numpy version and CPU, or None (with the reason) if none were recorded."""
    rec = GOLD['snow'].get(np.__version__)
    if rec is None:
        return None, 'no snow golden for numpy %s (recorded: %s)' % (np.__version__, sorted(GOLD['snow']))
    if rec['cpu_features'] is not None and rec['cpu_features'] != _cpu_features():
        return None, 'snow golden for numpy %s was recorded on a CPU with other SIMD features' % np.__version__
    return rec['cases'], ''


PC = gc.cloud()


def check(name, *vals):
    assert gc.digest(*vals) == GOLD['cases'][name], name


def check_snow(name, *vals):
    cases, why = snow_golden()
    if cases is None:
        pytest.skip(why)
    assert gc.digest(*vals) == cases[name], name


@pytest.mark.parametrize('sev', [1, 2, 3])
def test_points_reduce(sev):
    check('points_reduce_sev%d' % sev, PointsReductionInjector(severity=sev, seed=7)(PC))


@pytest.mark.parametrize('sev', [1, 2, 3])
def test_lidar_fog(sev):
    check('lidar_fog_sev%d' % sev, LidarFogInjector(severity=sev, seed=7)(PC))


@pytest.mark.slow
def test_lidar_snow(tmp_path, monkeypatch):
    monkeypatch.setenv('COOPFAULTBENCH_CACHE_DIR', os.environ.get('COOPFAULTBENCH_CACHE_DIR', str(tmp_path)))
    if snow_golden()[0] is None:
        pytest.skip(snow_golden()[1])
    inj = LidarSnowInjector(severity=1, seed=7, mount_height=1.9, verbose=False)
    check_snow('lidar_snow_sev1', inj(PC), inj.last_flags)


def test_cooperative_and_dropout():
    pe = PoseErrorInjector(seed=7, sigma_xy=0.4, sigma_heading=0.4)
    err = pe.sample_error()
    check('pose_error', np.array([err[k] for k in sorted(err)]), pe.perturb_matrix(gc.rot_z(30.0), error=err))
    check('agent_drop', np.array([AgentDropInjector(p_drop=0.5, seed=s)._lost('650') for s in range(200)]))
    check('missing_mask', np.array([bernoulli_mask(0.5, np.random.default_rng(s)) for s in range(200)]))
    check('latency', np.array([CommLatencyInjector(mu_delay=0.3, sigma_jitter=0.05, seed=s).stale_index('650', 50)
                               for s in range(50)]))
    mm = MissingModalityInjector(p_drop_rgb=0.5, p_drop_lidar=0.5, seed=11)
    outs = [mm.inject(gc.image(), PC) for _ in range(5)]
    check('missing_modality', *[x for o in outs for x in (o['image'], o['points'])])
    tm = TemporalMisalignmentInjector(mu_delay=0.2, sigma_jitter=0.05, seed=3)
    check('temporal', np.array([tm.stale_index(10) for _ in range(20)]))


def test_camera_side_own_code():
    img = gc.image()
    for kind in ('dirt', 'scratch', 'crack'):
        o, _ = SensorOcclusionInjector(OcclusionConfig(kind=kind, seed=5, texture_path=''))(img)
        check('occlusion_%s' % kind, o)
    ce = CalibrationErrorInjector(sigma_focal_px=5.0, sigma_principal_px=2.0, sigma_translation_m=0.05,
                                  sigma_rotation_deg=0.5, seed=9)
    k = np.array([[500.0, 0, 64], [0, 500.0, 48], [0, 0, 1]])
    cerr = ce.sample_error()
    check('calibration', np.array([cerr[x] if np.isscalar(cerr[x]) else np.sum(cerr[x]) for x in sorted(cerr)],
                                  dtype=float), ce.perturb_intrinsics(k, cerr), ce.perturb_extrinsics(np.eye(4), cerr))


@needs_camera
def test_camera_corruptions():
    from coopfaultbench.injectors import BrightnessInjector, CameraFogInjector, CameraSnowInjector, DarknessInjector
    img = gc.image()
    for name, cls in (('brightness', BrightnessInjector), ('darkness', DarknessInjector),
                      ('camera_fog', CameraFogInjector), ('camera_snow', CameraSnowInjector)):
        check('camera_%s_sev2' % name, cls(severity=2, seed=7)(img))


@pytest.mark.parametrize('name', ['pose_error_0.4', 'agent_drop_0.50', 'missing_lidar_0.50', 'points_reduce_sev2',
                                  'lidar_fog_sev2'])
@pytest.mark.parametrize('idx', [0, 123])
def test_apply_sample_faults(name, idx):
    assert gc.apply_case(name, idx) == GOLD['cases']['apply_%s_idx%d' % (name, idx)]


@pytest.mark.slow
@pytest.mark.parametrize('idx', [0, 123])
def test_apply_snow(idx, tmp_path, monkeypatch):
    monkeypatch.setenv('COOPFAULTBENCH_CACHE_DIR', os.environ.get('COOPFAULTBENCH_CACHE_DIR', str(tmp_path)))
    cases, why = snow_golden()
    if cases is None:
        pytest.skip(why)
    assert gc.apply_case('lidar_snow_sev1', idx) == cases['apply_lidar_snow_sev1_idx%d' % idx]


def test_golden_file_is_not_vacuous():
    assert len(GOLD['cases']) == 30 and not any('lidar_snow' in k for k in GOLD['cases'])
    # one legitimate coincidence: the same collaborators are dropped at frames 0 and 123
    assert len({v['sha256'] for v in GOLD['cases'].values()}) == 29
    assert GOLD['cases']['apply_agent_drop_0.50_idx0'] == GOLD['cases']['apply_agent_drop_0.50_idx123']
    assert set(GOLD['snow']) == {'1.21.6', '2.5.1'}
    a, b = GOLD['snow']['1.21.6']['cases'], GOLD['snow']['2.5.1']['cases']
    assert set(a) == set(b) == {'lidar_snow_sev1', 'apply_lidar_snow_sev1_idx0', 'apply_lidar_snow_sev1_idx123'}
    assert all(a[k] != b[k] for k in a)       # the documented numpy dependence of the snow output
