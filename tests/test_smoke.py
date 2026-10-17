# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Release gate G2: one injector per family on a tiny synthetic point cloud / image."""
import numpy as np

import golden_cases as gc
from conftest import needs_camera
from coopfaultbench.faults import FaultSpec, apply_sample_faults
from coopfaultbench.injectors import (LidarFogInjector, MissingModalityInjector, PointsReductionInjector,
                                      PoseErrorInjector, SensorOcclusionInjector, OcclusionConfig,
                                      TemporalMisalignmentInjector)

PC = gc.cloud(n=800)


def test_cooperative_family_pose_error_moves_collaborators_only():
    s = gc.sample()
    before = {a: s.agents[a].pose.copy() for a in s.agents}
    apply_sample_faults(s, FaultSpec(seed=7, pose_error={'sigma_xy': 0.4, 'sigma_heading': 0.4}), 0)
    moved = {a for a in s.agents if not np.array_equal(s.agents[a].pose, before[a])}
    assert moved == {'650', '659'}
    err = PoseErrorInjector(seed=1, sigma_xy=0.2, sigma_heading=0.2).sample_error()
    assert len(err) >= 2 and any(v != 0 for v in err.values())


def test_lidar_family_points_reduce_and_fog():
    out = PointsReductionInjector(severity=2, seed=3)(PC)
    assert out.shape == (round(len(PC) * 0.2), 4)
    fog = LidarFogInjector(severity=3, seed=3)(PC)
    assert fog.shape == PC.shape and fog.dtype == np.float32
    assert not np.array_equal(fog, PC) and np.all(np.isfinite(fog))


def test_dropout_family_missing_modality():
    img = gc.image()
    outs = [MissingModalityInjector(p_drop_rgb=1.0, p_drop_lidar=1.0, seed=s).inject(img, PC) for s in range(3)]
    for o in outs:
        assert not np.any(o['image']) and len(o['points']) == 0


def test_camera_family_own_injectors():
    img = gc.image()
    occluded, info = SensorOcclusionInjector(OcclusionConfig(kind='dirt', seed=1))(img)
    assert occluded.shape == img.shape and not np.array_equal(occluded, img)
    k, delta = TemporalMisalignmentInjector(mu_delay=0.3, sigma_jitter=0.0, seed=1).stale_index(10)
    assert k == 7 and delta == 3


@needs_camera
def test_camera_family_corruptions():
    from coopfaultbench.injectors import BrightnessInjector
    img = gc.image()
    out = BrightnessInjector(severity=3, seed=1)(img)
    assert out.shape == img.shape and out.dtype == np.uint8 and out.mean() > img.mean()
