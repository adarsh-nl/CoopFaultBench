# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""The OpenCOOD integration imports without OpenCOOD and drives a dataset class through its hooks."""
import json
import os

import numpy as np

import golden_cases as gc
from coopfaultbench.faults import FaultSpec
from coopfaultbench.opencood.adapter import OpenCOODAdapter, wrapper_fingerprint
from coopfaultbench.opencood.faulty_dataset import make_faulty_dataset


class _Base(object):
    """Stands in for an OpenCOOD dataset class (only what the hooks touch)."""

    def __init__(self, *args, **kwargs):
        pass


def test_imports_and_fingerprint():
    fp = wrapper_fingerprint()
    assert len(fp['adapter_sha256']) == 16
    assert OpenCOODAdapter(ns='opencood').ns == 'opencood'


def test_sample_stage_hook_matches_apply_sample_faults():
    spec = FaultSpec(seed=1234, pose_error={'sigma_xy': 0.4, 'sigma_heading': 0.4})
    ds = make_faulty_dataset(_Base, spec)()
    s = gc.sample()
    ds._fi_apply(s, 123)
    gold = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'injector_golden.json')))
    assert gc.digest(*gc.sample_digest(s)) == gold['cases']['apply_pose_error_0.4_idx123']   # the original code's output


def test_latency_hook_uses_the_scenario_of_the_frame():
    spec = FaultSpec(seed=1234, latency={'mu_delay': 0.3, 'sigma_jitter': 0.0})
    ds = make_faulty_dataset(_Base, spec)()
    ds.len_record = [10, 25, 40]
    ds.scenario_database = {0: {'1': {'ego': True}, '2': {'ego': False}},
                            1: {'5': {'ego': True}, '6': {'ego': False}, '7': {'ego': False}},
                            2: {'9': {'ego': True}}}
    ds._fi_build_delay_map(12)                  # frame 12 is the 3rd frame of scenario 1
    assert ds._fi_delays == [0, 2, 2]           # ego 0; 300 ms clamped to ts_index 2
    ds._fi_build_delay_map(30)
    assert ds._fi_delays == [0]


from conftest import needs_opencood  # noqa: E402


def _base_data(rng):
    from opencood.utils.transformation_utils import x1_to_x2
    poses = {641: [10.0, 5.0, 1.9, 0.0, 30.0, 0.0], 650: [30.0, -4.0, 1.9, 0.5, -60.0, 1.0]}
    out = {}
    for cav, pose in poses.items():
        out[cav] = {'ego': cav == 641, 'time_delay': 0,
                    'params': {'lidar_pose': pose, 'ego_speed': 3.0, 'vehicles': {},
                               'transformation_matrix': np.asarray(x1_to_x2(pose, poses[641])),
                               'gt_transformation_matrix': np.eye(4), 'spatial_correction_matrix': np.eye(4)},
                    'lidar_np': rng.random((50, 4)).astype(np.float32)}
    return out


@needs_opencood
def test_adapter_round_trip_is_exact_without_faults():
    rng = np.random.default_rng(0)
    base = _base_data(rng)
    before = {k: (v['params']['transformation_matrix'].copy(), v['lidar_np'].copy()) for k, v in base.items()}
    ad = OpenCOODAdapter()
    out = ad.from_canonical(ad.to_canonical(base, 7), base)
    assert list(out) == [641, 650]
    for k, (t, pts) in before.items():
        assert np.array_equal(out[k]['params']['transformation_matrix'], t)
        assert np.array_equal(out[k]['lidar_np'], pts)


@needs_opencood
def test_adapter_composes_pose_error_onto_the_baked_matrix():
    rng = np.random.default_rng(1)
    base = _base_data(rng)
    t0 = base[650]['params']['transformation_matrix'].copy()
    ad = OpenCOODAdapter()
    s = ad.to_canonical(base, 7)
    delta = np.eye(4)
    delta[0, 3] = 0.5
    s.agents['650'].pose = s.agents['650'].pose @ delta          # sender's pose error in its own frame
    out = ad.from_canonical(s, base)
    assert np.allclose(out[650]['params']['transformation_matrix'], t0 @ delta)
    assert np.array_equal(out[641]['params']['transformation_matrix'], base[641]['params']['transformation_matrix'])


def test_adapter_takes_x_to_world_from_its_namespace(tmp_path, monkeypatch):
    """With ns='v2xvit' the adapter must not need OpenCOOD (V2X-ViT's code base ships the same module)."""
    pkg = tmp_path / 'fakecodebase' / 'utils'
    pkg.mkdir(parents=True)
    (tmp_path / 'fakecodebase' / '__init__.py').write_text('')
    (pkg / '__init__.py').write_text('')
    (pkg / 'transformation_utils.py').write_text(
        'import numpy as np\n\ndef x_to_world(pose):\n    m = np.eye(4)\n    m[0, 3] = 42.0\n    return m\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    ad = OpenCOODAdapter(ns='fakecodebase')
    assert ad.x_to_world([0, 0, 0, 0, 0, 0])[0, 3] == 42.0


def test_latency_hook_pops_one_delay_per_agent_ego_first():
    spec = FaultSpec(seed=1234, latency={'mu_delay': 0.3, 'sigma_jitter': 0.0})
    ds = make_faulty_dataset(_Base, spec)()
    ds.len_record = [10, 25]
    ds.scenario_database = {0: {'1': {'ego': True}, '2': {'ego': False}},
                            1: {'5': {'ego': True}, '6': {'ego': False}}}
    ds._fi_build_delay_map(9)                      # ts_index 9 >= 3: the full 300 ms (3 frames)
    assert ds.time_delay_calculation(True) == 0    # ego first, always 0
    assert ds.time_delay_calculation(False) == 3
    import pytest
    with pytest.raises(AssertionError):
        ds.time_delay_calculation(False)           # the map is exhausted: one entry per agent
    ds._fi_delays = [2]
    with pytest.raises(AssertionError):
        ds.time_delay_calculation(True)            # a non-zero delay for the ego is a misaligned map
