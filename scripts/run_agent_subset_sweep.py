#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Agent-subset sweep with dual-reference scoring (needs OpenCOOD, a checkpoint and the dataset).

For every frame, evaluate EVERY agent subset containing the ego -- ego-only up
to the full roster -- and score each configuration twice:

    union  G(A')  the union over the evaluated subset, recomputed per subset
                  (what OpenCOOD does; the agent-dependent reference)
    fixed  G(A)   the union over the frame's full COM_RANGE-surviving roster,
                  held constant across every subset of that frame

This tool measures; it decides nothing. Its rows feed
coopfaultbench.evaluation.agent_drop (scripts/resample_agent_drop.py).

REALIZED ROSTER, NOT FORCED ROSTER -- applied in all three places.
COM_RANGE filtering happens inside __getitem__, AFTER retrieve_base_data, so a
forced subset member can still be dropped by range; 25.5% of clean frames lose
an agent that way. Keying anything off the forced subset would contaminate a
quarter of the grid. So:

  (1) cache key      (frame, realized_roster). Two different forced subsets
                     that collapse to the same realized roster share ONE
                     forward pass and one scored row.
  (2) k assignment   a forced k=3 whose third member is out of range is a
                     realized k=2 datapoint and is binned there.
  (3) bootstrap      duplicate realized rosters within a frame are emitted
                     once, with the forced subsets that produced them listed.
                     Emitting them repeatedly would silently inflate effective
                     N in the scenario-level bootstrap.

The realized roster is READ FROM THE RELEASED PATH, not recomputed: a recorder
wraps get_item_single_car, which the dataset calls exactly once per surviving
CAV. Reimplementing the distance test would be a second, divergable copy.

EXCLUSIONS. A configuration that cannot be evaluated is RECORDED with a
machine-readable reason, never dropped silently and never fatal -- at 15,694
configs a raise would kill the run.

RESUME. Per-scenario JSONL is appended as each frame completes and the
detection cache is written per frame, so keep --out_dir on persistent storage;
after an interruption, frames already present are skipped; the
continuation is bit-identical because the RNG contract is per-(frame, agent).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

from coopfaultbench.opencood.sweep_utils import (  # noqa: E402
    forced, install_rng_and_recorder, sha_dets, tag_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--out_dir', required=True,
                    help='persistent directory (resume state lives here)')
    ap.add_argument('--scenarios', required=True,
                    help='comma-separated scenario names, this chunk')
    ap.add_argument('--ns', default='opencood')
    ap.add_argument('--limit_frames', type=int, default=0,
                    help='smoke only: cap frames per scenario')
    a = ap.parse_args()

    torch = importlib.import_module('torch')
    train_utils = importlib.import_module(a.ns + '.tools.train_utils')
    inf_mod = importlib.import_module(a.ns + '.tools.inference_utils')
    eval_utils = importlib.import_module(a.ns + '.utils.eval_utils')
    yaml_utils = importlib.import_module(a.ns + '.hypes_yaml.yaml_utils')
    build_ds = importlib.import_module(a.ns + '.data_utils.datasets')

    hypes = yaml_utils.load_yaml(os.path.join(a.model_dir, 'config.yaml'), None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = build_ds.build_dataset(copy.deepcopy(hypes), visualize=False,
                                train=False)
    # SEPARATE, NEVER-WRAPPED object: supplies the fixed reference.
    ds_clean = build_ds.build_dataset(copy.deepcopy(hypes), visualize=False,
                                      train=False)
    tag_ids(ds)

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(a.model_dir, model)
    model.to(device).eval()

    # frame -> scenario, built the way the dataset builds it
    # so the two orderings cannot drift.
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
        det_dir = os.path.join(a.out_dir, 'dets', scen)
        os.makedirs(det_dir, exist_ok=True)

        done = set()
        if os.path.exists(rows_path):
            for ln in open(rows_path):
                try:
                    done.add(json.loads(ln)['frame'])
                except Exception:
                    pass
        print('[subset-sweep] %s: %d frames, %d already done -> resuming'
              % (scen, len(idxs), len(done)), flush=True)

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
            others = sorted(c for c in roster if c != ego)

            # Fixed reference: once per frame, from the UNWRAPPED dataset,
            # through the RELEASED GT path.
            #
            # It must come out in the SAME representation the scorer consumes:
            # a torch tensor of CORNER boxes (n, 8, 3), which is what
            # generate_gt_bbx returns and what inference_intermediate_fusion
            # hands back as the union reference. An earlier version passed
            # `object_bbx_center[mask == 1]` -- a numpy array in CENTRE form
            # (n, 7) -- and eval_utils.caluclate_tp_fp died on `.is_cuda`.
            # Two representations for the two references would not have been
            # comparable even had it run.
            try:
                bfix = ds_clean.collate_batch_test([ds_clean[fr]])
                bfix = train_utils.to_device(bfix, device)
                gt_fixed = ds_clean.post_processor.generate_gt_bbx(bfix)
                gt_fixed_n = 0 if gt_fixed is None else int(gt_fixed.shape[0])
            except Exception as e:                       # noqa: BLE001
                fh.write(json.dumps({'frame': fr, 'scenario': scen,
                                     'excluded': 'fixed_reference_error',
                                     'detail': str(e)[:200]}) + '\n')
                fh.flush()
                continue

            by_realized = {}
            excl = []
            for r in range(len(others) + 1):
                for combo in combinations(others, r):
                    keep = {ego} | set(combo)
                    undo_f = forced(ds, keep)
                    undo_r, seen = install_rng_and_recorder(ds, fr)
                    try:
                        batch = ds.collate_batch_test([ds[fr]])
                        batch = train_utils.to_device(batch, device)
                        with torch.no_grad():
                            pb, ps, gt_u = \
                                inf_mod.inference_intermediate_fusion(
                                    batch, model, ds)
                    except Exception as e:               # noqa: BLE001
                        # recorded, never silent, never fatal.
                        excl.append({'forced': sorted(keep),
                                     'reason': 'forward_error',
                                     'detail': str(e)[:200]})
                        continue
                    finally:
                        undo_r()
                        undo_f()

                    realized = tuple(sorted(set(seen)))
                    if not realized:
                        excl.append({'forced': sorted(keep),
                                     'reason': 'keep_list_empty_after_range_filter'})
                        continue
                    if ego not in realized:
                        excl.append({'forced': sorted(keep),
                                     'reason': 'ego_dropped'})
                        continue

                    # (1) cache key IS the realized roster: a forced subset
                    # that collapses onto an already-evaluated realized roster
                    # reuses that forward pass and adds no duplicate row.
                    if realized in by_realized:
                        by_realized[realized]['forced_variants'].append(
                            sorted(keep))
                        continue

                    # FAIL FAST: the two references must be the same
                    # representation, or every dAP is meaningless. Checked on
                    # the first scored config, not after a long run.
                    if gt_u is not None and gt_fixed is not None:
                        if type(gt_u) is not type(gt_fixed) or \
                                gt_u.shape[1:] != gt_fixed.shape[1:]:
                            print('FATAL: reference representations differ -- '
                                  'union %s%s vs fixed %s%s'
                                  % (type(gt_u).__name__, tuple(gt_u.shape[1:]),
                                     type(gt_fixed).__name__,
                                     tuple(gt_fixed.shape[1:])), flush=True)
                            return 1

                    boxes = (np.zeros((0, 8, 3), np.float32) if pb is None
                             else pb.detach().cpu().numpy())
                    scores = (np.zeros((0,), np.float32) if ps is None
                              else ps.detach().cpu().numpy())
                    gt_union_n = 0 if gt_u is None else int(gt_u.shape[0])

                    # dual scoring: the SAME detections against both references
                    st_u = {thr: dict(tp=[], fp=[], gt=0, score=[]) for thr in (0.5, 0.7)}
                    st_f = copy.deepcopy(st_u)
                    for iou in (0.5, 0.7):
                        eval_utils.caluclate_tp_fp(pb, ps, gt_u, st_u, iou)
                        eval_utils.caluclate_tp_fp(pb, ps, gt_fixed,
                                                    st_f, iou)

                    by_realized[realized] = {
                        'realized': list(realized),
                        'k': len(realized),          # (2) k from REALIZED
                        'forced_variants': [sorted(keep)],
                        'n_gt_union': gt_union_n,
                        'n_gt_fixed': gt_fixed_n,
                        'sha256': sha_dets(boxes, scores),
                        'n_det': int(boxes.shape[0]),
                        'tp_union': {str(i): [int(v) for v in st_u[i]['tp']]
                                     for i in (0.5, 0.7)},
                        'fp_union': {str(i): [int(v) for v in st_u[i]['fp']]
                                     for i in (0.5, 0.7)},
                        'gtn_union': {str(i): int(st_u[i]['gt'])
                                      for i in (0.5, 0.7)},
                        'tp_fixed': {str(i): [int(v) for v in st_f[i]['tp']]
                                     for i in (0.5, 0.7)},
                        'fp_fixed': {str(i): [int(v) for v in st_f[i]['fp']]
                                     for i in (0.5, 0.7)},
                        'gtn_fixed': {str(i): int(st_f[i]['gt'])
                                      for i in (0.5, 0.7)},
                        'score': {str(i): [float(v) for v in st_u[i]['score']]
                                  for i in (0.5, 0.7)},
                    }
                    np.savez_compressed(
                        os.path.join(det_dir, 'f%06d_%s.npz'
                                     % (fr, '-'.join(realized))),
                        boxes=boxes, scores=scores)

            # (3) one row per DISTINCT realized roster -- duplicates would
            # inflate effective N in the scenario-level bootstrap.
            fh.write(json.dumps({
                'frame': fr, 'scenario': scen, 'N': len(roster),
                'ego': ego, 'roster': roster,
                'n_forced_enumerated': 2 ** len(others),
                'n_distinct_realized': len(by_realized),
                'configs': list(by_realized.values()),
                'exclusions': excl,
                'wall_s': round(time.time() - t0, 3)}) + '\n')
            fh.flush()
            os.fsync(fh.fileno())      # survive an interrupted job
        fh.close()
        print('[subset-sweep] %s complete' % scen, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
