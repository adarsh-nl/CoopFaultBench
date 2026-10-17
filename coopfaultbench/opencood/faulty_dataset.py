# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""
faulty_dataset.py
-----------------
Turn an OpenCOOD dataset class into a fault-injecting one.

``make_faulty_dataset(base_cls, spec)`` returns a subclass of ``base_cls`` with
two hooks. It takes the class **as an argument**, so the same code works against
OpenCOOD and against V2X-ViT's code base (namespace ``v2xvit``).

Two hooks, not one
------------------
Every injector but one runs on the assembled ``CooperativeSample``. CommLatency
cannot: it decides *which file to read*, upstream of assembly. So:

* **hook 1** -- ``time_delay_calculation``: pre-assembly, frame selection.
  Overriding OpenCOOD's own seam keeps ``time_delay`` as a genuine model input
  (V2X-ViT's prior-encoding branch consumes it).
* **hook 2** -- ``retrieve_base_data``: post-assembly, everything else.

Unit note for hook 1: OpenCOOD's ``time_delay_calculation`` returns an integer
in units of **100 ms** (``time_delay = time_delay // 100``), then clamps against
``timestamp_index``. ``CommLatencyInjector.stale_index`` is **frame**-based. At
the 10 Hz of OPV2V/V2XSet, 1 frame == 1 unit, so the map is identity -- but the
conversion is named here so nobody later passes seconds.

Reproducibility under DataLoader workers
----------------------------------------
``opencood/tools/inference.py`` runs the loader with ``num_workers=16``, so
``retrieve_base_data`` executes in 16 forked processes. Any RNG state held on
the dataset object is therefore **per-worker**, and stateful injectors
(``AgentDropInjector``'s Gilbert-Elliott state, ``PoseErrorInjector``'s single
``default_rng``) would not reproduce across worker counts.

So every injector here is constructed **per call**, seeded from
``SeedSequence(base_seed, spawn_key=(idx, crc32(agent_id), stage))``. Nothing
accumulates. A run gives identical results at any ``num_workers`` including 0,
and every log row is re-derivable from ``(base_seed, idx, agent_id)``.

Known gap, stated rather than hidden: **bursty** (Gilbert-Elliott) AgentDrop is
inherently sequential and cannot be expressed this way. Only i.i.d. AgentDrop is
supported here; a bursty variant needs the chain advanced ``timestamp_index``
steps from a per-scenario seed, which is O(k) per sample. ``p_drop`` i.i.d. --
what the paper's protocol uses -- has no such issue.
"""

import bisect

import numpy as np

from ..faults import FaultSpec, InjectionLog, STAGE, derive_seed  # noqa: F401 (re-exported)
from .adapter import OpenCOODAdapter


def make_faulty_dataset(base_cls, spec, adapter=None):
    """Return a subclass of ``base_cls`` that injects faults per ``spec``."""
    adapter = adapter or OpenCOODAdapter()
    from ..faults import apply_sample_faults, derive_seed as _seed

    class FaultyDataset(base_cls):

        fi_spec = spec
        fi_adapter = adapter

        def __init__(self, *args, **kwargs):
            super(FaultyDataset, self).__init__(*args, **kwargs)
            self._fi_log = InjectionLog(spec.log_dir)
            self._fi_delays = None
            self._fi_checked_modality = False
            if spec.missing_lidar and spec.missing_lidar.get('p_drop_rgb'):
                raise ValueError(
                    'p_drop_rgb must be 0: OPV2V/V2XSet through OpenCOOD are '
                    'LiDAR-only, retrieve_base_data never reads the PNGs')
            if spec.lidar_snow:
                # Generate the per-severity snowflake particle files ONCE,
                # here in the parent process. Left to first use they would be
                # generated inside DataLoader workers -- 16 forked processes
                # racing to write the same 64 .npy files.
                from ..injectors.snow_particles import ensure_particle_files
                ensure_particle_files(spec.lidar_snow.get('severity', 2),
                                      verbose=False)

        # ── hook 1: pre-assembly frame selection ────────────────────────

        def time_delay_calculation(self, ego_flag):
            """
            ``time_delay_calculation(self, ego_flag)`` carries neither ``idx``
            nor ``cav_id``, so the delay map is built up front in
            ``retrieve_base_data`` (which we override anyway) and consumed here
            in the same ``scenario_database.items()`` order the caller loops
            in. One entry per cav INCLUDING the ego, popped unconditionally --
            an early return for the ego would leave its entry in the queue and
            hand the ego's 0 to the first cav.
            """
            if spec.latency is None or self._fi_delays is None:
                return super(FaultyDataset, self).time_delay_calculation(ego_flag)
            if not self._fi_delays:
                raise AssertionError('delay map exhausted: the cav loop ran '
                                     'more times than the map was built for')
            delay = self._fi_delays.pop(0)
            if ego_flag and delay != 0:
                raise AssertionError('ego delay must be 0, map is misaligned')
            return delay

        def _fi_build_delay_map(self, idx):
            from ..injectors.communication import CommLatencyInjector
            # len_record holds the cumulative frame count at the end of each
            # scenario; the scenario of frame idx is the first one whose end
            # lies beyond idx.
            scenario_index = bisect.bisect_right(self.len_record, idx)
            if scenario_index >= len(self.len_record):
                raise IndexError('frame %d is beyond the last scenario' % idx)
            db = self.scenario_database[scenario_index]
            ts_index = idx if scenario_index == 0 \
                else idx - self.len_record[scenario_index - 1]

            delays = []
            for cav_id, content in db.items():
                if content['ego']:
                    delays.append(0)
                    continue
                s = _seed(spec.seed, idx, cav_id, 'latency')
                inj = CommLatencyInjector(seed=s, **spec.latency)
                # stale_index returns (k_stale, delta_frames); frames ==
                # OpenCOOD's 100 ms units at the 10 Hz of OPV2V/V2XSet.
                # basedataset additionally clamps against timestamp_index,
                # so a scene-start frame cannot borrow from before the scene.
                k_stale, delta = inj.stale_index(str(cav_id), ts_index)
                delay = int(max(0, ts_index - k_stale))
                delays.append(delay)
                if delay > 0:
                    self._fi_log.write(
                        idx=idx, agent_id=str(cav_id), is_ego=0,
                        stage='latency', seed=s,
                        detail='delay_frames=%d;drawn=%d;ts_index=%d'
                               % (delay, delta, ts_index))
            self._fi_delays = delays

        # ── hook 2: post-assembly sample faults ─────────────────────────

        def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
            if spec.latency is not None:
                self._fi_build_delay_map(idx)
            base = super(FaultyDataset, self).retrieve_base_data(
                idx, cur_ego_pose_flag=cur_ego_pose_flag)
            if spec.is_null:
                # Still round-trip: the null path must exercise exactly the
                # same adapter code the faulty path does, or a clean-vs-official
                # comparison proves nothing about the faulty run.
                sample = self.fi_adapter.to_canonical(base, idx)
                return self.fi_adapter.from_canonical(sample, base)

            sample = self.fi_adapter.to_canonical(base, idx)

            if not self._fi_checked_modality:
                self.fi_adapter.assert_modality(sample, 'lidar')
                self._fi_checked_modality = True

            ego_pose_before = sample.ego.pose.copy()
            self._fi_apply(sample, idx)

            if not np.array_equal(sample.ego.pose, ego_pose_before):
                raise AssertionError(
                    'ego pose was perturbed. GT is generated with the ego '
                    'pose as reference_lidar_pose, so this would move the '
                    'ground truth and the AP delta would not be attributable '
                    'to cooperative misalignment.')
            return self.fi_adapter.from_canonical(sample, base)

        # ── the sample-stage faults, all stateless per (idx, agent) ─────

        def _fi_apply(self, sample, idx):
            # The weather cache is reused only within one frame: the weather
            # output of an (agent, stage) is deterministic per frame, and a
            # driver that calls this twice per frame (once per scored arm)
            # would otherwise recompute it.
            if getattr(self, '_fi_weather_cache_idx', None) != idx:
                self._fi_weather_cache = {}
                self._fi_weather_cache_idx = idx
            apply_sample_faults(sample, spec, idx, log=self._fi_log,
                                weather_cache=self._fi_weather_cache,
                                latency_in_loader=True)   # hook 1 applies it

    FaultyDataset.__name__ = 'Faulty' + base_cls.__name__
    return FaultyDataset
