# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Fault injectors for cooperative (multi-agent) 3D perception.

Every injector works on plain arrays and poses: an (N, C) LiDAR array, an (H, W, 3) RGB image, a
(4, 4) pose matrix, or a :class:`coopfaultbench.sample.CooperativeSample`.

Cooperative / V2X faults (applied to collaborators, never to the ego):
    PoseErrorInjector         localisation error on shared poses
    CommLatencyInjector       transmission latency (stale frames)
    AgentDropInjector         collaborator dropout (i.i.d.)
    BandwidthLimitInjector    bandwidth-limited point sharing (not used in the paper)

Sensor and environment faults:
    MissingModalityInjector   sensor dropout (Bernoulli-gated)
    PointsReductionInjector   uniform LiDAR point dropout (MultiCorrupt, MIT)
    LidarFogInjector          LiDAR fog      -- uses lidar_weather/, CC BY-NC 4.0: NON-COMMERCIAL ONLY
    LidarSnowInjector         LiDAR snowfall -- uses lidar_weather/, CC BY-NC 4.0: NON-COMMERCIAL ONLY
    TemporalMisalignmentInjector, SensorOcclusionInjector, CalibrationErrorInjector (camera-side)
    BrightnessInjector, DarknessInjector, CameraFogInjector, CameraSnowInjector
        (camera corruptions; Apache-2.0 backend in _vendor/imagecorruptions; need the `camera` extra)

Which agents a fault touches in the paper's protocol (see docs/injectors.md): points_reduce, lidar_fog
and lidar_snow hit ALL agents including the ego (sensor/environment faults); pose_error, latency,
agent_drop and missing_lidar hit collaborators only.
"""

from .pose_error import PoseErrorInjector
from .communication import CommLatencyInjector, AgentDropInjector, BandwidthLimitInjector
from .missing_modality import MissingModalityInjector, drop_image, drop_points, bernoulli_mask
from .temporal_misalignment import (TemporalMisalignmentInjector, sample_index_shift,
                                    physical_displacement)
from .sensor_occlusion import SensorOcclusionInjector, OcclusionConfig
from .calibration import CalibrationErrorInjector, rotation_from_axis_angle
from .points_reduce import PointsReductionInjector
from .lidar_fog import LidarFogInjector
from .lidar_snow import LidarSnowInjector

_CAMERA = ('BrightnessInjector', 'DarknessInjector', 'CameraFogInjector', 'CameraSnowInjector')

__all__ = [
    'PoseErrorInjector', 'CommLatencyInjector', 'AgentDropInjector', 'BandwidthLimitInjector',
    'MissingModalityInjector', 'drop_image', 'drop_points', 'bernoulli_mask',
    'TemporalMisalignmentInjector', 'sample_index_shift', 'physical_displacement',
    'SensorOcclusionInjector', 'OcclusionConfig', 'CalibrationErrorInjector', 'rotation_from_axis_angle',
    'PointsReductionInjector', 'LidarFogInjector', 'LidarSnowInjector',
] + ['injector_fingerprint', 'NON_COMMERCIAL_INJECTORS']
# The camera corruptions (_CAMERA) are importable by name but not part of `import *`, so a star import
# works without the optional camera dependencies.

#: Injectors whose code path runs CC BY-NC 4.0 code (coopfaultbench/injectors/lidar_weather/).
NON_COMMERCIAL_INJECTORS = ('LidarFogInjector', 'LidarSnowInjector')


def __getattr__(name):
    """Camera corruptions are imported on first use; they need the optional `camera` extra.

    A missing dependency raises ImportError naming the extra. There are no silent stubs.
    """
    if name in _CAMERA:
        try:
            from . import camera_corruptions
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError('%s needs the optional camera dependencies: '
                              'pip install "coopfaultbench[camera]" (%s)' % (name, e)) from e
        return getattr(camera_corruptions, name)
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def injector_fingerprint():
    """Content hash of the injector sources (all .py files of this package and its subpackages).

    Results files record it so that a change to any injector is visible. Renaming and splitting the
    files for the licence boundary changed the fingerprint: the paper's fault-inference cells record
    'f222966a809bfeab' and its five-family fault-sweep rows '59ad2698eac56722' (computed differently by
    that driver) for the original code. Behavioural equivalence with that code is shown by
    tests/test_injector_golden.py, not by the fingerprint.
    """
    import hashlib
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    names = []
    for dp, dns, fns in os.walk(here):
        dns[:] = sorted(d for d in dns if d != '__pycache__')
        for n in sorted(fns):
            if n.endswith('.py'):
                names.append(os.path.relpath(os.path.join(dp, n), here).replace(os.sep, '/'))
    per, h = {}, hashlib.sha256()
    for n in sorted(names):
        with open(os.path.join(here, n), 'rb') as fh:
            b = fh.read()
        h.update(n.encode() + b'\0' + b)
        per[n] = hashlib.sha256(b).hexdigest()[:8]
    return {'injector_sha256': h.hexdigest()[:16], 'injector_files_sha256': per}
