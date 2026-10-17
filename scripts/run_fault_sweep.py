#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Five-family fault sweep with dual-reference scoring (needs OpenCOOD, a checkpoint and the dataset).

For every frame of the requested scenarios, under one fault condition (``--fault`` x ``--tier`` from
coopfaultbench/configs/fault_grid.yaml), two arms are evaluated:

    individual  k = 1          the ego alone
    cp          k = N_realized the full communication-range-surviving roster

and each arm's detections are scored against the union reference (of the realised subset) and the fixed
reference (the unperturbed full roster, from a dataset the fault never touches). One JSON row per frame is
written to ``<out_dir>/rows_<scenario>.jsonl``; the run resumes per frame. scripts/analyze_fault_sweep.py
turns the rows of all conditions into the fault-sweep payload.

The five families are pose_error, latency, points_reduce, lidar_fog and lidar_snow (agent_drop and
missing_modality are excluded: they are a second, stochastic control over the agent set this sweep already
controls). lidar_fog and lidar_snow run CC BY-NC 4.0 code: non-commercial use only.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import time

import numpy as np

from coopfaultbench.configs import condition_spec
from coopfaultbench.faults import FaultSpec
from coopfaultbench.injectors import injector_fingerprint
from coopfaultbench.opencood.adapter import OpenCOODAdapter
from coopfaultbench.opencood.faulty_dataset import make_faulty_dataset
from coopfaultbench.opencood.sweep_utils import forced, install_rng_and_recorder, sha_dets, tag_ids

FAMILIES = ('pose_error', 'latency', 'points_reduce', 'lidar_fog', 'lidar_snow')
# collaborator-only faults vs faults on every agent (the paper's protocol)
SCOPE = {'pose_error': 'non-ego', 'latency': 'non-ego',
         'points_reduce': 'all', 'lidar_fog': 'all', 'lidar_snow': 'all'}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--scenarios', required=True)
    ap.add_argument('--fault', required=True, choices=sorted(FAMILIES))
    ap.add_argument('--tier', required=True,
                    choices=('mild', 'moderate', 'severe'))
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--ns', default='opencood')
    ap.add_argument('--limit_frames', type=int, default=0)
    ap.add_argument('--skip_ego_arm', action='store_true',
                    help='collaborator-only faults (pose_error, latency): the k=1 arm is '
                         'identical to clean by construction and is taken from the '
                         'agent-subset sweep rows instead.')
    a = ap.parse_args()

    torch = importlib.import_module('torch')
    train_utils = importlib.import_module(a.ns + '.tools.train_utils')
    inf_mod = importlib.import_module(a.ns + '.tools.inference_utils')
    eval_utils = importlib.import_module(a.ns + '.utils.eval_utils')
    yaml_utils = importlib.import_module(a.ns + '.hypes_yaml.yaml_utils')
    build_ds = importlib.import_module(a.ns + '.data_utils.datasets')

    hypes = yaml_utils.load_yaml(os.path.join(a.model_dir, 'config.yaml'), None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cond = condition_spec(a.fault, a.tier)
    scope = SCOPE[a.fault]
    # No scope kwarg is injected: pose_error and latency are collaborator-only
    # by construction. SCOPE only decides whether the ego arm can be reused.
    spec = FaultSpec(seed=a.seed, **cond)
    fp = injector_fingerprint()['injector_sha256']
    print('[fault-sweep] fault=%s tier=%s scope=%s spec=%s injector_sha=%s'
          % (a.fault, a.tier, scope, json.dumps(cond), fp), flush=True)

    base_cls = build_ds.build_dataset(copy.deepcopy(hypes), visualize=False,
                                      train=False).__class__
    Faulty = make_faulty_dataset(base_cls, spec, adapter=OpenCOODAdapter(ns=a.ns))
    ds = Faulty(params=copy.deepcopy(hypes), visualize=False, train=False)
    # UNFAULTED reference dataset: never wrapped, never injected.
    ds_clean = build_ds.build_dataset(copy.deepcopy(hypes), visualize=False,
                                      train=False)
    tag_ids(ds)

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(a.model_dir, model)
    model.to(device).eval()

    root = hypes.get('validate_dir') or hypes['root_dir']
    scen_names = sorted(x for x in os.listdir(root)
                        if os.path.isdir(os.path.join(root, x)))
    scen_of, prev = [], 0
    for i, upper in enumerate(ds_clean.len_record):
        scen_of.extend([scen_names[i]] * (upper - prev))
        prev = upper

    want = [s.strip() for s in a.scenarios.split(',') if s.strip()]
    os.makedirs(a.out_dir, exist_ok=True)

    for scen in want:
        idxs = [i for i, s in enumerate(scen_of) if s == scen]
        if a.limit_frames:
            idxs = idxs[:a.limit_frames]
        rows_path = os.path.join(a.out_dir, 'rows_%s.jsonl' % scen)
        done = set()
        if os.path.exists(rows_path):
            for ln in open(rows_path):
                try:
                    done.add(json.loads(ln)['frame'])
                except Exception:
                    pass
        print('[fault-sweep] %s: %d frames, %d done -> resuming' % (scen, len(idxs),
                                                           len(done)),
              flush=True)
        fh = open(rows_path, 'a')
        for fr in idxs:
            if fr in done:
                continue
            t0 = time.time()
            bd = ds.retrieve_base_data(fr)
            roster = [str(c) for c in bd]
            ego = next((str(c) for c, v in bd.items() if v['ego']), None)
            if ego is None:
                fh.write(json.dumps({'frame': fr, 'scenario': scen,
                                     'excluded': 'no_ego_in_frame'}) + '\n')
                fh.flush()
                continue

            try:
                bfix = ds_clean.collate_batch_test([ds_clean[fr]])
                bfix = train_utils.to_device(bfix, device)
                gt_fixed = ds_clean.post_processor.generate_gt_bbx(bfix)
                gt_fixed_n = 0 if gt_fixed is None else int(gt_fixed.shape[0])
            except Exception as e:                        # noqa: BLE001
                fh.write(json.dumps({'frame': fr, 'scenario': scen,
                                     'excluded': 'fixed_reference_error',
                                     'detail': str(e)[:200]}) + '\n')
                fh.flush()
                continue

            arms = [] if a.skip_ego_arm else [('individual', {ego})]
            arms.append(('cp', set(roster)))

            out_cfgs, excl = [], []
            for arm, keep in arms:
                undo_f = forced(ds, keep)
                undo_r, seen = install_rng_and_recorder(ds, fr)
                try:
                    batch = ds.collate_batch_test([ds[fr]])
                    batch = train_utils.to_device(batch, device)
                    with torch.no_grad():
                        pb, ps, gt_u = inf_mod.inference_intermediate_fusion(
                            batch, model, ds)
                except Exception as e:                    # noqa: BLE001
                    excl.append({'arm': arm, 'reason': 'forward_error',
                                 'detail': str(e)[:200]})
                    continue
                finally:
                    undo_r()
                    undo_f()

                realized = tuple(sorted(set(seen)))
                if not realized or ego not in realized:
                    excl.append({'arm': arm,
                                 'reason': 'keep_list_empty_after_range_filter'
                                 if not realized else 'ego_dropped'})
                    continue

                boxes = (np.zeros((0, 8, 3), np.float32) if pb is None
                         else pb.detach().cpu().numpy())
                scores = (np.zeros((0,), np.float32) if ps is None
                          else ps.detach().cpu().numpy())
                st_u = {thr: dict(tp=[], fp=[], gt=0, score=[]) for thr in (0.5, 0.7)}
                st_f = copy.deepcopy(st_u)
                for iou in (0.5, 0.7):
                    eval_utils.caluclate_tp_fp(pb, ps, gt_u, st_u, iou)
                    eval_utils.caluclate_tp_fp(pb, ps, gt_fixed, st_f, iou)

                out_cfgs.append({
                    'arm': arm, 'realized': list(realized),
                    'k': len(realized), 'forced': sorted(keep),
                    'n_gt_union': 0 if gt_u is None else int(gt_u.shape[0]),
                    'n_gt_fixed': gt_fixed_n,
                    'sha256': sha_dets(boxes, scores),
                    'n_det': int(boxes.shape[0]),
                    'tp_union': {str(i): [int(v) for v in st_u[i]['tp']] for i in (0.5, 0.7)},
                    'fp_union': {str(i): [int(v) for v in st_u[i]['fp']] for i in (0.5, 0.7)},
                    'gtn_union': {str(i): int(st_u[i]['gt']) for i in (0.5, 0.7)},
                    'tp_fixed': {str(i): [int(v) for v in st_f[i]['tp']] for i in (0.5, 0.7)},
                    'fp_fixed': {str(i): [int(v) for v in st_f[i]['fp']] for i in (0.5, 0.7)},
                    'gtn_fixed': {str(i): int(st_f[i]['gt']) for i in (0.5, 0.7)},
                    'score': {str(i): [float(v) for v in st_u[i]['score']] for i in (0.5, 0.7)},
                })

            fh.write(json.dumps({
                'frame': fr, 'scenario': scen, 'N': len(roster), 'ego': ego,
                'roster': roster, 'fault': a.fault, 'tier': a.tier,
                'scope': scope, 'seed': a.seed, 'injector_sha256': fp,
                'spec': cond, 'configs': out_cfgs, 'exclusions': excl,
                'wall_s': round(time.time() - t0, 3)}) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
        fh.close()
        print('[fault-sweep] %s complete' % scen, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
