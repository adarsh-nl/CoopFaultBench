# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Synthetic cases whose outputs were recorded with the ORIGINAL injector code (tests/data/injector_golden.json).

The same cases, run through this package, must give byte-identical outputs.
"""
import hashlib
import json

import numpy as np

from coopfaultbench.faults import FaultSpec, apply_sample_faults
from coopfaultbench.sample import AgentFrame, CooperativeSample


def digest(*vals):
    h = hashlib.sha256()
    shapes = []
    for v in vals:
        a = np.ascontiguousarray(np.asarray(v))
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        shapes.append(list(a.shape))
    return {'sha256': h.hexdigest(), 'shape': shapes}


def cloud(seed=20261016, n=6000):
    g = np.random.default_rng(seed)
    r = g.uniform(3, 70, n // 2)
    th = g.uniform(-np.pi, np.pi, n // 2)
    ground = np.c_[r * np.cos(th), r * np.sin(th), -1.9 + g.normal(0, 0.02, n // 2)]
    m = n - n // 2
    cx = g.choice([8.0, 15.0, 25.0], m)
    cy = g.choice([-4.0, 3.0, 10.0], m)
    obj = np.c_[cx + g.uniform(-2, 2, m), cy + g.uniform(-1, 1, m), g.uniform(-1.9, -0.3, m)]
    xyz = np.vstack([ground, obj])
    inten = g.uniform(0, 1, n)
    return np.c_[xyz, inten].astype(np.float32)


def image(seed=7, h=96, w=128):
    g = np.random.default_rng(seed)
    base = g.integers(0, 256, (h // 8, w // 8, 3)).astype(np.float64)
    return np.kron(base, np.ones((8, 8, 1))).astype(np.uint8)


def rot_z(deg):
    a = np.radians(deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return T


def sample():
    s = CooperativeSample(frame_index=123, ego_id='641')
    for i, (aid, dx, yaw) in enumerate((('641', 0.0, 0.0), ('650', 20.0, 30.0), ('659', -15.0, -60.0))):
        T = rot_z(yaw)
        T[:3, 3] = (dx, 5.0 * i, 0.0)
        s.agents[aid] = AgentFrame(agent_id=aid, is_ego=(aid == '641'), pose=T, lidar=cloud(seed=100 + i, n=3000))
    return s


def sample_digest(s):
    vals = []
    for aid in sorted(s.agents):
        a = s.agents[aid]
        vals += [np.frombuffer(aid.encode(), dtype=np.uint8), a.pose, np.asarray(a.lidar)]
    vals.append(np.frombuffer(json.dumps(s.meta, sort_keys=True).encode(), dtype=np.uint8))
    return vals


CONDITIONS = {'pose_error_0.4': {'pose_error': {'sigma_xy': 0.4, 'sigma_heading': 0.4}},
              'agent_drop_0.50': {'agent_drop': {'p_drop': 0.5}},
              'missing_lidar_0.50': {'missing_lidar': {'p_drop_lidar': 0.5}},
              'points_reduce_sev2': {'points_reduce': {'severity': 2}},
              'lidar_fog_sev2': {'lidar_fog': {'severity': 2}},
              'lidar_snow_sev1': {'lidar_snow': {'severity': 1, 'mount_height': 1.9}}}


def apply_case(name, idx):
    kw = dict(CONDITIONS[name])
    kw['seed'] = 1234
    return digest(*sample_digest(apply_sample_faults(sample(), FaultSpec(**kw), idx)))
