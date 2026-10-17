# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Fault conditions and their application to one cooperative sample.

A :class:`FaultSpec` describes one fault condition (for example "pose error, 0.4 m / 0.4 deg").
:func:`apply_sample_faults` applies it to a :class:`~coopfaultbench.sample.CooperativeSample`.
Both work without OpenCOOD; :mod:`coopfaultbench.opencood` uses them inside OpenCOOD's data loading.

Reproducibility
---------------
Every injector is constructed per call and seeded from
``SeedSequence(base_seed, spawn_key=(frame_index, crc32(agent_id), stage))``. Nothing accumulates
between calls, so results do not depend on the order in which frames are loaded or on the number of
DataLoader workers, and every draw can be re-derived from ``(seed, frame_index, agent_id, stage)``.
The stage ordinals in :data:`STAGE` keep the sub-streams of one (frame, agent) independent; they are
part of the reproducibility contract and must not be renumbered.

Which agents a fault touches
----------------------------
* pose_error, agent_drop, missing_lidar, latency: collaborators only. The ego is never perturbed
  (latency is applied by the data loader, see coopfaultbench.opencood).
* points_reduce, lidar_fog, lidar_snow: every agent INCLUDING the ego by default (``scope='all'``),
  because they model sensor or environment degradation. ``scope='non-ego'`` restricts them to
  collaborators. The paper's results use the default.

Only i.i.d. agent drop is supported: bursty (Gilbert-Elliott) drop is sequential and cannot be derived
per frame.
"""
import csv
import os
import zlib

import numpy as np

from .injectors.communication import AgentDropInjector
from .injectors.missing_modality import bernoulli_mask, drop_points
from .injectors.pose_error import PoseErrorInjector

#: Stage ordinals of the per-(frame, agent) seed derivation (do not renumber).
STAGE = {'pose_error': 0, 'agent_drop': 1, 'missing_lidar': 2,
         'points_reduce': 3, 'latency': 4, 'lidar_fog': 5, 'lidar_snow': 6}


def derive_seed(base_seed, frame_index, agent_id, stage):
    """Reproducible 32-bit seed for one (frame, agent, stage) draw.

    ``zlib.crc32`` rather than ``hash()``: Python's string hash is salted per process.
    """
    key = (int(frame_index), int(zlib.crc32(str(agent_id).encode())), int(STAGE[stage]))
    return int(np.random.SeedSequence(entropy=int(base_seed),
                                      spawn_key=key).generate_state(1)[0])


class FaultSpec:
    """Declarative description of one fault condition.

    Parameters
    ----------
    pose_error    : dict or None  kwargs for ``PoseErrorInjector`` (no ``seed``).
    agent_drop    : dict or None  ``{'p_drop': float}``, i.i.d. only.
    missing_lidar : dict or None  ``{'p_drop_lidar': float}``.
    points_reduce : dict or None  ``{'severity': 1|2|3}`` plus optional ``'scope': 'all'|'non-ego'``.
    latency       : dict or None  kwargs for ``CommLatencyInjector`` (applied by the data loader).
    lidar_fog     : dict or None  kwargs for ``LidarFogInjector`` (no ``seed``), optional ``'scope'``.
    lidar_snow    : dict or None  kwargs for ``LidarSnowInjector`` (no ``seed``), optional ``'scope'``.
                    On OPV2V/V2XSet pass ``mount_height=1.9`` (the CARLA LiDAR mount).
    seed          : int           base seed; every draw derives from it.
    log_dir       : str or None   where per-process injection logs (CSV) are written.

    lidar_fog and lidar_snow run CC BY-NC 4.0 code (coopfaultbench/injectors/lidar_weather/):
    non-commercial use only.
    """

    FIELDS = ('pose_error', 'agent_drop', 'missing_lidar', 'points_reduce', 'latency',
              'lidar_fog', 'lidar_snow')

    def __init__(self, pose_error=None, agent_drop=None, missing_lidar=None,
                 points_reduce=None, latency=None, lidar_fog=None,
                 lidar_snow=None, seed=0, log_dir=None):
        self.pose_error = pose_error
        self.agent_drop = agent_drop
        self.missing_lidar = missing_lidar
        self.points_reduce = points_reduce
        self.latency = latency
        self.lidar_fog = lidar_fog
        self.lidar_snow = lidar_snow
        self.seed = seed
        self.log_dir = log_dir

    @property
    def is_null(self):
        return not any(getattr(self, f) for f in self.FIELDS)

    def as_dict(self):
        d = {f: getattr(self, f) for f in self.FIELDS}
        d['seed'] = self.seed
        return d


class InjectionLog:
    """One CSV per process (``injection_summary.<pid>.csv``); concatenate afterwards."""

    FIELDS = ('idx', 'agent_id', 'is_ego', 'stage', 'seed', 'detail')

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self._fh = None
        self._writer = None
        self._pid = None

    def write(self, **row):
        if self.log_dir is None:
            return
        if self._fh is None or self._pid != os.getpid():
            self._pid = os.getpid()
            path = os.path.join(self.log_dir, 'injection_summary.%d.csv' % self._pid)
            new = not os.path.exists(path)
            self._fh = open(path, 'a')
            self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
            if new:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()


def _fmt(err):
    return ';'.join('%s=%.6f' % (k, v) for k, v in sorted(err.items()))


def apply_sample_faults(sample, spec, idx, log=None, weather_cache=None, latency_in_loader=False):
    """Apply the sample-stage faults of ``spec`` to ``sample`` in place (frame index ``idx``).

    Latency is not a sample-stage fault: it decides WHICH frame a collaborator's data is read from, so it
    must act in the data loader (coopfaultbench.opencood.faulty_dataset does this; for your own loader use
    ``CommLatencyInjector.stale_index``). A spec with ``latency`` therefore raises ValueError here unless the
    caller states ``latency_in_loader=True``.

    Order: agent_drop, pose_error, missing_lidar, lidar_fog, lidar_snow, points_reduce.
    ``weather_cache``: optional dict reused across calls for the SAME frame; the weather output of an
    (agent, stage) is deterministic per frame, so repeated calls on one frame may reuse it. The cache is
    keyed by (frame, agent, stage) and checked against a CRC of the input points.
    """
    if spec.latency and not latency_in_loader:
        raise ValueError('latency selects which frame is read and must be applied by the data loader '
                         '(coopfaultbench.opencood.faulty_dataset, or CommLatencyInjector.stale_index); '
                         'pass latency_in_loader=True if your loader already applied it')
    log = log if log is not None else InjectionLog(None)
    cache = weather_cache if weather_cache is not None else {}

    if spec.agent_drop:
        for aid in [a for a in list(sample.agents) if not sample.agents[a].is_ego]:
            s = derive_seed(spec.seed, idx, aid, 'agent_drop')
            inj = AgentDropInjector(seed=s, **spec.agent_drop)
            if inj._lost(aid):
                del sample.agents[aid]
                sample.meta.setdefault('dropped_agents', []).append(aid)
                log.write(idx=idx, agent_id=aid, is_ego=0, stage='agent_drop', seed=s,
                          detail='dropped')

    if spec.pose_error:
        for aid, agent in sample.agents.items():
            if agent.is_ego:
                continue
            s = derive_seed(spec.seed, idx, aid, 'pose_error')
            inj = PoseErrorInjector(seed=s, **spec.pose_error)
            err = inj.sample_error()
            agent.pose = inj.perturb_matrix(agent.pose, error=err)
            agent.faults['pose_error'] = err
            log.write(idx=idx, agent_id=aid, is_ego=0, stage='pose_error', seed=s, detail=_fmt(err))

    if spec.missing_lidar:
        p = spec.missing_lidar['p_drop_lidar']
        for aid, agent in sample.agents.items():
            if agent.is_ego:
                continue
            s = derive_seed(spec.seed, idx, aid, 'missing_lidar')
            keep = bernoulli_mask(p, np.random.default_rng(s))
            agent.faults['missing_lidar'] = {'m_lidar': int(keep)}
            if not keep:
                agent.lidar = drop_points(agent.lidar)
                log.write(idx=idx, agent_id=aid, is_ego=0, stage='missing_lidar', seed=s,
                          detail='lidar_dropped')

    for stage_name in ('lidar_fog', 'lidar_snow'):
        cfg = getattr(spec, stage_name)
        if not cfg:
            continue
        cfg = dict(cfg)
        scope = cfg.pop('scope', 'all')
        if stage_name == 'lidar_fog':
            from .injectors.lidar_fog import LidarFogInjector as _Weather
        else:
            from .injectors.lidar_snow import LidarSnowInjector as _Weather
            cfg.setdefault('verbose', False)
        for aid, agent in sample.agents.items():
            if scope == 'non-ego' and agent.is_ego:
                continue
            if agent.lidar is None or not len(agent.lidar):
                continue          # e.g. already dropped by missing_lidar
            s = derive_seed(spec.seed, idx, aid, stage_name)
            before = agent.lidar
            cache_key = (idx, aid, stage_name)
            before_crc = zlib.crc32(np.ascontiguousarray(before))
            cached = cache.get(cache_key)
            if cached is not None:
                cached_crc, cached_after = cached
                if cached_crc != before_crc:
                    raise AssertionError('weather cache: input CRC differs for %r; refusing to reuse'
                                         % (cache_key,))
                after = cached_after.copy()
            else:
                after = _Weather(seed=s, **cfg)(before)
                after_store = np.array(after, copy=True)
                after_store.setflags(write=False)
                cache[cache_key] = (before_crc, after_store)
            agent.lidar = after
            detail = ('pts=%d->%d;meanI=%.4f->%.4f'
                      % (len(before), len(after),
                         float(np.asarray(before)[:, 3].mean()),
                         float(np.asarray(after)[:, 3].mean())))
            agent.faults[stage_name] = detail
            log.write(idx=idx, agent_id=aid, is_ego=int(agent.is_ego), stage=stage_name, seed=s,
                      detail=detail)

    if spec.points_reduce:
        from .injectors.points_reduce import PointsReductionInjector
        cfg = dict(spec.points_reduce)
        scope = cfg.pop('scope', 'all')
        for aid, agent in sample.agents.items():
            if scope == 'non-ego' and agent.is_ego:
                continue
            if agent.lidar is None or not len(agent.lidar):
                continue
            s = derive_seed(spec.seed, idx, aid, 'points_reduce')
            before = agent.lidar
            after = np.ascontiguousarray(PointsReductionInjector(seed=s, **cfg)(before),
                                         dtype=np.float32)
            agent.lidar = after
            detail = 'pts=%d->%d' % (len(before), len(after))
            agent.faults['points_reduce'] = detail
            log.write(idx=idx, agent_id=aid, is_ego=int(agent.is_ego), stage='points_reduce', seed=s,
                      detail=detail)
    return sample
