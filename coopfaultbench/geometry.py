# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Pose <-> matrix helpers for the CARLA pose convention used by OPV2V and V2XSet.

A pose is ``[x, y, z, roll, yaw, pitch]``: metres and degrees, in that order, as stored in the
OPV2V/V2XSet yaml files and consumed by OpenCOOD. The matrix is the 4x4 rigid transform from the
pose's frame to the world frame.

CARLA uses Unreal Engine's left-handed rotation convention. Expressed with right-handed elementary
rotations about the world axes, the rotation is

    R = Rz(yaw) @ Ry(-pitch) @ Rx(-roll)

(yaw about z, then pitch and roll with their signs flipped by the handedness change).

Modules that already need OpenCOOD (coopfaultbench.opencood) use OpenCOOD's own
``opencood.utils.transformation_utils.x_to_world`` instead. This module exists for code that must work
without OpenCOOD (synthetic tests, CPU tools). tests/test_geometry.py checks it against OpenCOOD's
function whenever OpenCOOD is importable.
"""
import numpy as np


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose_to_matrix(pose):
    """4x4 pose-frame-to-world transform of a CARLA pose ``[x, y, z, roll, yaw, pitch]`` (deg)."""
    x, y, z, roll, yaw, pitch = [float(v) for v in pose[:6]]
    rot = _rot_z(np.radians(yaw)) @ _rot_y(-np.radians(pitch)) @ _rot_x(-np.radians(roll))
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = (x, y, z)
    return out


def relative_transform(pose_from, pose_to):
    """4x4 transform taking points in ``pose_from``'s frame into ``pose_to``'s frame."""
    return np.linalg.inv(pose_to_matrix(pose_to)) @ pose_to_matrix(pose_from)


def matrix_to_pose(matrix):
    """Inverse of :func:`pose_to_matrix` for |pitch| < 90 deg: returns ``[x, y, z, roll, yaw, pitch]``."""
    m = np.asarray(matrix, dtype=np.float64)
    r = m[:3, :3]
    # From R = Rz(yaw) Ry(-pitch) Rx(-roll): R[2,0] = sin(pitch), R[2,1] = -cos(pitch) sin(roll),
    # R[2,2] = cos(pitch) cos(roll), R[1,0] = sin(yaw) cos(pitch), R[0,0] = cos(yaw) cos(pitch).
    pitch = np.arcsin(np.clip(r[2, 0], -1.0, 1.0))
    roll = np.arctan2(-r[2, 1], r[2, 2])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return np.array([m[0, 3], m[1, 3], m[2, 3],
                     np.degrees(roll), np.degrees(yaw), np.degrees(pitch)])
