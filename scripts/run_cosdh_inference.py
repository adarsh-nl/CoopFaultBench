#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""CoSDH inference with fault injection (needs the CoSDH code base, its checkpoint and OPV2V).

CoSDH (https://github.com/Xu2729/CoSDH) is an OpenCOOD fork that differs at every hook point, so it has its own
driver and adapter (coopfaultbench.opencood.cosdh). Two interception points, and only two:

1. The base dataset class. CoSDH's ``build_dataset`` resolves the base class by NAME in its datasets module
   (``<DATASET>BaseDataset``). For the duration of the dataset construction that name is bound to the faulty
   subclass (``cosdh.make_faulty_base``); CoSDH's own ``build_dataset`` then runs unchanged.
2. The pose fault. It is installed on the fusion dataset's ``add_noise_data_dict``, AFTER CoSDH copies
   ``lidar_pose`` into ``lidar_pose_clean`` (``cosdh.make_pose_noise_hook``).

Everything downstream -- inference, post-processing, matching, AP -- is CoSDH's unmodified ``inference.main()``.

    PYTHONPATH=/path/to/CoSDH python scripts/run_cosdh_inference.py --model_dir DIR --fi_out OUT \\
        [--fi_condition points_reduce/severe | --fi_spec '{"lidar_fog": {"severity": 2}}']

``--fi_condition official`` (the default) runs CoSDH without any wrapper; ``clean`` runs the wrapper with no
fault. Agent drop and missing modality are not supported for CoSDH (all such cells failed in the original runs).
"""
import argparse
import importlib
import json
import os
import sys

from coopfaultbench.configs import load_fault_grid
from coopfaultbench.faults import FaultSpec, InjectionLog
from coopfaultbench.injectors import injector_fingerprint
from coopfaultbench.opencood.adapter import wrapper_fingerprint
from coopfaultbench.opencood.cosdh import cosdh_fingerprint, make_faulty_base, make_pose_noise_hook

CONDITIONS = {'official': None, 'clean': dict()}
for _inj, _cfg in load_fault_grid()['injectors'].items():
    if _inj in ('agent_drop', 'missing_modality'):
        continue
    for _tier in ('mild', 'moderate', 'severe'):
        CONDITIONS['%s/%s' % (_inj, _tier)] = dict(_cfg[_tier]['spec'])

_AP = {}


def capture_ap(eval_utils, oc_inference):
    """inference.main() computes AP but returns nothing: record it (the original still computes and dumps)."""
    orig = eval_utils.eval_final_results

    def wrapped(result_stat, save_path, infer_info=None):
        out = orig(result_stat, save_path, infer_info)
        try:
            _AP['ap_30'], _AP['ap_50'], _AP['ap_70'] = float(out[0]), float(out[1]), float(out[2])
        except Exception:                                   # noqa: BLE001
            pass
        _AP['n_gt'] = int(result_stat[0.5].get('gt', -1))
        return out
    eval_utils.eval_final_results = wrapped
    oc_inference.eval_utils.eval_final_results = wrapped


def install(spec, log, ds_mod, oc_inference, ilf, pose_utils):
    """Bind the faulty base class for dataset construction; hook the pose fault."""
    orig_build = oc_inference.build_dataset

    def build_dataset(dataset_cfg, visualize=False, train=True):
        name = dataset_cfg['fusion']['dataset'].upper() + 'BaseDataset'
        base = getattr(ds_mod, name)
        setattr(ds_mod, name, make_faulty_base(base, spec, log))
        try:
            print('[cosdh] base dataset %s wrapped' % name, flush=True)
            return orig_build(dataset_cfg, visualize=visualize, train=train)
        finally:
            setattr(ds_mod, name, base)

    oc_inference.build_dataset = build_dataset
    if spec.pose_error is not None:
        ilf.add_noise_data_dict = make_pose_noise_hook(pose_utils.add_noise_data_dict, spec, log)
        print('[cosdh] pose hook installed after add_noise_data_dict', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--fusion_method', default='intermediatelate')
    ap.add_argument('--fi_condition', default='official', choices=sorted(CONDITIONS))
    ap.add_argument('--fi_spec', default=None, help='FaultSpec kwargs as JSON (overrides --fi_condition)')
    ap.add_argument('--fi_seed', type=int, default=1234)
    ap.add_argument('--fi_out', required=True)
    ap.add_argument('--epoch', type=int, default=-1)
    a = ap.parse_args()

    opencood = importlib.import_module('opencood')
    ds_mod = importlib.import_module('opencood.data_utils.datasets')
    # Running CoSDH's checkpoint against vanilla OpenCOOD would be silent and wrong. The CoSDH fork has the
    # intermediate-late fusion dataset; vanilla OpenCOOD does not.
    if not hasattr(ds_mod, 'getIntermediatelateFusionDataset'):
        raise SystemExit('the imported "opencood" (%s) is not the CoSDH fork; put the CoSDH checkout first on '
                         'PYTHONPATH' % os.path.dirname(opencood.__file__))
    ilf = importlib.import_module('opencood.data_utils.datasets.intermediate_late_fusion_dataset')
    oc_inference = importlib.import_module('opencood.tools.inference')
    pose_utils = importlib.import_module('opencood.utils.pose_utils')
    eval_utils = importlib.import_module('opencood.utils.eval_utils')

    os.makedirs(a.fi_out, exist_ok=True)
    cond = json.loads(a.fi_spec) if a.fi_spec is not None else CONDITIONS[a.fi_condition]
    spec = None if cond is None else FaultSpec(seed=a.fi_seed, **cond)
    if spec is None:
        print('[cosdh] UNWRAPPED: official pipeline, no interception', flush=True)
    else:
        if spec.agent_drop or spec.missing_lidar:
            raise SystemExit('agent_drop / missing_lidar are not supported for CoSDH')
        if spec.lidar_snow:     # generate the particle files once, here, not inside DataLoader workers
            from coopfaultbench.injectors.snow_particles import ensure_particle_files
            ensure_particle_files(spec.lidar_snow.get('severity', 2), verbose=False)
        log_dir = os.path.join(a.fi_out, 'injection')
        os.makedirs(log_dir, exist_ok=True)
        install(spec, InjectionLog(log_dir), ds_mod, oc_inference, ilf, pose_utils)

    capture_ap(eval_utils, oc_inference)
    sys.argv = ['inference.py', '--model_dir', a.model_dir, '--fusion_method', a.fusion_method,
                '--epoch', str(a.epoch)]
    rc = oc_inference.main()
    out = dict(_AP)
    out.update(seed=a.fi_seed, spec_json=a.fi_spec, condition=a.fi_condition,
               spec=spec.as_dict() if spec else None, fusion_method=a.fusion_method)
    out.update(wrapper_fingerprint())
    out.update(injector_fingerprint())
    out.update(cosdh_fingerprint())
    with open(os.path.join(a.fi_out, 'fi_result.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('[cosdh] AP@0.5 %r  AP@0.7 %r  -> %s' % (_AP.get('ap_50'), _AP.get('ap_70'), a.fi_out), flush=True)
    return 0 if rc is None else rc


if __name__ == '__main__':
    sys.exit(main())
