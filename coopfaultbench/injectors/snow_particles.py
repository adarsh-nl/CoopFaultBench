# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Seeded generation of the snowflake particle files used by LidarSnowInjector.

For each severity, 64 particle sets (one per LiDAR channel of the HDL-64 model) are sampled with
Hahner et al.'s dart_throwing sampler (lidar_weather/snowflake_sampling.py, CC BY-NC 4.0) at
R_0 = 80 m with the 'gunn' distribution, channel i with numpy.random.default_rng(seed + i).
Files are named gunn_{rain_rate}_{occupancy}_{i}.npy, i = 1..64, as the simulation expects.

The files are OUTPUT of the CC BY-NC sampler and are treated as covered by that licence.
They are written to a cache directory, never into the package:
    $COOPFAULTBENCH_CACHE_DIR/snow_particles/training/snowflakes/npy/   (if the variable is set)
    ~/.cache/coopfaultbench/snow_particles/training/snowflakes/npy/      (otherwise)
Generation takes a few minutes per severity, once per machine.
"""
import os
import numpy as np
from pathlib import Path

from .lidar_weather.snowflake_sampling import dart_throwing


def particle_root():
    """Root passed to the snow simulation as root_path."""
    base = os.environ.get('COOPFAULTBENCH_CACHE_DIR')
    base = Path(base) if base else Path.home() / '.cache' / 'coopfaultbench'
    return base / 'snow_particles'


def particle_dir():
    """Directory holding the gunn_*.npy files (the layout the simulation reads)."""
    return particle_root() / 'training' / 'snowflakes' / 'npy'


# Severity -> (snowfall_rate [mm/h], terminal_velocity [m/s]); MultiCorrupt lidar.py
SEVERITY_SNOWFALL = {1: (0.5, 1.2), 2: (2.5, 1.6), 3: (1.5, 0.4)}
NUM_CHANNELS = 64          # HDL-64 sensor model used by the snow simulation
R_0 = 80.0                 # sampling-domain radius in metres (Hahner sampling.py)


def particle_file_prefix(severity: int) -> str:
    """The exact prefix MultiCorrupt's simulate_snow builds for this severity."""
    from .lidar_weather.snow_simulation import snowfall_rate_to_rainfall_rate, compute_occupancy
    snowfall_rate, terminal_velocity = (float(v) for v in SEVERITY_SNOWFALL[severity])
    rain_rate = snowfall_rate_to_rainfall_rate(snowfall_rate, terminal_velocity)
    occupancy = compute_occupancy(snowfall_rate, terminal_velocity)
    return f'gunn_{rain_rate}_{occupancy}'


def ensure_particle_files(severity: int, npy_dir=None, seed: int = 42,
                          show_progressbar: bool = False, verbose: bool = True):
    """Generate (if missing) the 64 particle files for a severity; returns the prefix.

    Files are written to `npy_dir` (default: particle_dir()). Existing files are
    kept, so generation is a one-time cost per machine.
    """
    from .lidar_weather.snow_simulation import snowfall_rate_to_rainfall_rate, compute_occupancy
    npy_dir = Path(npy_dir) if npy_dir is not None else particle_dir()
    npy_dir.mkdir(parents=True, exist_ok=True)

    snowfall_rate, terminal_velocity = (float(v) for v in SEVERITY_SNOWFALL[severity])
    rain_rate = snowfall_rate_to_rainfall_rate(snowfall_rate, terminal_velocity)
    occupancy = compute_occupancy(snowfall_rate, terminal_velocity)
    prefix = f'gunn_{rain_rate}_{occupancy}'

    missing = [i for i in range(1, NUM_CHANNELS + 1)
               if not (npy_dir / f'{prefix}_{i}.npy').exists()]
    if missing and verbose:
        print(f'[snow_particles] severity {severity}: generating {len(missing)} '
              f'particle files (one-time; Hahner dart_throwing, R_0={R_0} m) ...')
    for i in missing:
        rng = np.random.default_rng(seed + i)
        particles = dart_throwing(occupancy_ratio=occupancy, precipitation_rate=rain_rate,
                                  R_0=R_0, rng=rng, distribution='gunn',
                                  show_progessbar=show_progressbar)
        final = npy_dir / f'{prefix}_{i}.npy'
        tmp = npy_dir / f'.{prefix}_{i}.{os.getpid()}.tmp.npy'
        np.save(str(tmp), particles)
        os.replace(str(tmp), str(final))            # atomic: no truncated file if interrupted
        if verbose and (i % 16 == 0 or i == missing[-1]):
            print(f'[snow_particles]   {prefix}_{i}.npy  ({len(particles)} flakes)')
    return prefix
