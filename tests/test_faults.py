# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
import bisect

import numpy as np
import pytest

import golden_cases as gc
from coopfaultbench.configs import condition_spec, load_fault_grid
from coopfaultbench.faults import STAGE, FaultSpec, apply_sample_faults, derive_seed


def test_stage_ordinals_are_frozen():
    assert STAGE == {'pose_error': 0, 'agent_drop': 1, 'missing_lidar': 2, 'points_reduce': 3, 'latency': 4,
                     'lidar_fog': 5, 'lidar_snow': 6}


def test_seed_derivation_is_stable_and_distinct():
    a = derive_seed(1234, 5, '650', 'pose_error')
    assert a == derive_seed(1234, 5, '650', 'pose_error')
    assert len({derive_seed(1234, 5, '650', s) for s in STAGE}) == len(STAGE)
    assert derive_seed(1234, 5, '650', 'pose_error') != derive_seed(1234, 6, '650', 'pose_error')


@pytest.mark.parametrize('spec', [{'pose_error': {'sigma_xy': 0.6, 'sigma_heading': 0.6}},
                                  {'agent_drop': {'p_drop': 0.75}},
                                  {'missing_lidar': {'p_drop_lidar': 0.75}}])
def test_link_faults_never_touch_the_ego(spec):
    s0 = gc.sample()
    for idx in range(10):
        s = gc.sample()
        apply_sample_faults(s, FaultSpec(seed=1234, **spec), idx)
        assert s.ego_id in s.agents
        assert np.array_equal(s.ego.pose, s0.ego.pose) and np.array_equal(s.ego.lidar, s0.ego.lidar)


def test_sensor_faults_hit_the_ego_by_default_and_scope_restricts():
    s = gc.sample()
    apply_sample_faults(s, FaultSpec(seed=1, points_reduce={'severity': 3}), 0)
    assert len(s.ego.lidar) < len(gc.sample().ego.lidar)
    s = gc.sample()
    apply_sample_faults(s, FaultSpec(seed=1, points_reduce={'severity': 3, 'scope': 'non-ego'}), 0)
    assert len(s.ego.lidar) == len(gc.sample().ego.lidar)
    assert all(len(a.lidar) < 3000 for aid, a in s.agents.items() if aid != s.ego_id)


def test_null_spec_changes_nothing():
    s = gc.sample()
    spec = FaultSpec(seed=1)
    assert spec.is_null
    apply_sample_faults(s, spec, 3)
    s0 = gc.sample()
    assert all(np.array_equal(s.agents[a].lidar, s0.agents[a].lidar) for a in s0.agents)


def test_scenario_lookup_by_bisect_equals_a_linear_scan():
    g = np.random.default_rng(0)
    for _ in range(200):
        record = list(np.cumsum(g.integers(1, 50, int(g.integers(1, 12)))))
        for idx in range(record[-1]):
            linear = next(i for i, end in enumerate(record) if idx < end)
            assert bisect.bisect_right(record, idx) == linear


def test_fault_grid():
    grid = load_fault_grid()
    assert grid['seed'] == 1234 and grid['tiers'] == ['mild', 'moderate', 'severe']
    assert set(grid['injectors']) == {'pose_error', 'latency', 'agent_drop', 'missing_modality', 'points_reduce',
                                      'lidar_fog', 'lidar_snow'}
    for inj in grid['injectors']:
        for tier in grid['tiers']:
            spec = FaultSpec(seed=grid['seed'], **condition_spec(inj, tier))
            assert not spec.is_null
    assert condition_spec('lidar_snow', 'severe') == {'lidar_snow': {'severity': 3, 'mount_height': 1.9}}


def test_latency_is_not_silently_ignored():
    with pytest.raises(ValueError):
        apply_sample_faults(gc.sample(), FaultSpec(seed=1, **condition_spec('latency', 'severe')), 0)
    s = apply_sample_faults(gc.sample(), FaultSpec(seed=1, **condition_spec('latency', 'severe')), 0,
                            latency_in_loader=True)
    assert set(s.agents) == {'641', '650', '659'}
