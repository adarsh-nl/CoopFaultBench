# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Running faults inside OpenCOOD (and V2X-ViT's and CoSDH's OpenCOOD-based code bases).

Everything here needs OpenCOOD at RUN time; importing this package does not. OpenCOOD is used as an
external dependency under its own licence and is not redistributed: install it yourself from
https://github.com/DerrickXuNu/OpenCOOD.

    adapter.OpenCOODAdapter          retrieve_base_data dict <-> CooperativeSample
    faulty_dataset.make_faulty_dataset(base_cls, spec)
                                     subclass of an OpenCOOD dataset class that injects faults
    cosdh.make_faulty_base / make_pose_noise_hook
                                     the same for the CoSDH fork (its hook points differ)
"""
