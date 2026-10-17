# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Packaged configuration: the paper's fault grid (fault_grid.yaml)."""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_fault_grid():
    """The paper's fault grid as a dict (see fault_grid.yaml)."""
    with open(os.path.join(HERE, 'fault_grid.yaml')) as fh:
        return yaml.safe_load(fh)


def condition_spec(injector, tier):
    """FaultSpec keyword arguments of one grid cell, e.g. ('lidar_fog', 'severe')."""
    grid = load_fault_grid()
    return dict(grid['injectors'][injector][tier]['spec'])
