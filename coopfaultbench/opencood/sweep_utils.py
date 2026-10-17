# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Helpers shared by the OpenCOOD sweep drivers (scripts/run_agent_subset_sweep.py, run_fault_sweep.py).

* the realised roster is RECORDED from OpenCOOD's own data path (``get_item_single_car`` is called once per
  agent that survives the communication-range filter), not recomputed;
* OpenCOOD's point shuffling draws from numpy's global random state; it is seeded per (frame, agent) here, so
  the same agent's points are shuffled identically in every evaluated subset;
* a subset of agents is forced by filtering ``retrieve_base_data``; the dataset that supplies the FIXED
  reference is never wrapped.
"""
import hashlib

import numpy as np


def agent_seed(frame_id, cav_id):
    return int.from_bytes(hashlib.blake2s(
        ('%s|%s' % (frame_id, cav_id)).encode(), digest_size=4).digest(), 'big')


def tag_ids(ds):
    """Carry cav_id into get_item_single_car, which is not given it."""
    orig = ds.retrieve_base_data

    def f(i, cur_ego_pose_flag=True):
        bd = orig(i, cur_ego_pose_flag)
        for cav_id, v in bd.items():
            v['_cfb_cav_id'] = str(cav_id)
        return bd
    ds.retrieve_base_data = f
    return orig


def install_rng_and_recorder(ds, frame_id):
    """Per-agent seeding AND realized-roster recording, in one wrapper.

    get_item_single_car is called once per CAV that survives the COM_RANGE
    filter, so the ids seen here ARE the realized roster -- taken from the
    released path rather than by re-deriving the distance test.
    """
    orig = ds.get_item_single_car
    seen = []

    def f(selected_cav_base, *a, **kw):
        cid = selected_cav_base.get('_cfb_cav_id', '?')
        seen.append(cid)
        np.random.seed(agent_seed(frame_id, cid))
        return orig(selected_cav_base, *a, **kw)
    ds.get_item_single_car = f

    def undo():
        ds.get_item_single_car = orig
    return undo, seen


def forced(ds, keep):
    """Force the participating subset. Returns ``undo``, which restores the dataset.

    NEVER installed on the unperturbed reference dataset: that
    would collapse `fixed` onto `union` and make every dAP vacuously zero.
    """
    orig = ds.retrieve_base_data

    def f(i, cur_ego_pose_flag=True):
        bd = orig(i, cur_ego_pose_flag)
        out = type(bd)()
        for cav_id, v in bd.items():
            if str(cav_id) in keep:
                out[cav_id] = v
        return out
    ds.retrieve_base_data = f

    def undo():
        ds.retrieve_base_data = orig
    return undo


def sha_dets(boxes, scores):
    return hashlib.sha256(np.ascontiguousarray(boxes).tobytes()
                          + np.ascontiguousarray(scores).tobytes()).hexdigest()
