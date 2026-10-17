# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
import numpy as np

from conftest import needs_opencood
from coopfaultbench.geometry import matrix_to_pose, pose_to_matrix, relative_transform


def poses(n=200, seed=0):
    g = np.random.default_rng(seed)
    return np.c_[g.uniform(-200, 200, (n, 3)), g.uniform(-30, 30, n), g.uniform(-180, 180, n), g.uniform(-30, 30, n)]


def test_rigid_and_invertible():
    for p in poses():
        m = pose_to_matrix(p)
        r = m[:3, :3]
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1.0)
        assert np.allclose(m[:3, 3], p[:3]) and np.allclose(m[3], [0, 0, 0, 1])
        back = matrix_to_pose(m)
        assert np.allclose(back, p, atol=1e-9)


def test_yaw_only_is_a_z_rotation():
    m = pose_to_matrix([1, 2, 3, 0, 90, 0])
    assert np.allclose(m[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)


def test_relative_transform_maps_frames():
    a, b = poses(2, seed=3)
    t = relative_transform(a, b)
    assert np.allclose(pose_to_matrix(b) @ t, pose_to_matrix(a))


@needs_opencood
def test_matches_opencood_x_to_world():
    from opencood.utils.transformation_utils import x_to_world
    for p in poses():
        assert np.allclose(pose_to_matrix(p), np.asarray(x_to_world(list(p))), rtol=0, atol=1e-9)
