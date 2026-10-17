# SPDX-License-Identifier: CC-BY-NC-4.0
# Part of coopfaultbench/injectors/lidar_weather/. Licensed under CC BY-NC 4.0
# (https://creativecommons.org/licenses/by-nc/4.0/), NOT under the repository's MIT licence.
# Non-commercial use only. See LICENSE and NOTICE in this directory.
# Copied from Martin Hahner et al., "LiDAR Snowfall Simulation" (CVPR 2022), directly from
#   https://github.com/SysCV/LiDAR_snow_sim tools/snowfall/sampling.py (upstream commit: UNRESOLVED),
#   whose header states __author__ = "Martin Hahner", __license__ = "CC BY-NC 4.0".
# Changes: tqdm progress-bar support made optional; nothing else.
"""Snowflake particle sampling (Hahner et al.): sekhon_srivastava, gunn_marshall, dart_throwing."""
import numpy as np

try:
    from tqdm import tqdm
except Exception:                                     # tqdm optional
    tqdm = None

PI = np.pi

# ============================ BEGIN VERBATIM (Hahner) ============================
def sekhon_srivastava(precipitation_rate: float) -> float:
    """
    :param precipitation_rate:  in mm/h
    :return:                    in 1/cm
    """
    # Determine rate parameter of distribution of snowflake diameters via formula of Sekhon and Srivastava (1970).
    return 22.9 * precipitation_rate ** -0.45


def gunn_marshall(precipitation_rate: float) -> float:
    """
    :param precipitation_rate:  in mm/h
    :return:                    in 1/cm
    """
    # Determine rate parameter of distribution of snowflake diameters via formula of Marshall and Gunn (1958).
    return 25.5 * precipitation_rate ** -0.48


def dart_throwing(occupancy_ratio: float,
                  precipitation_rate: float,
                  R_0: float,
                  rng: np.random.Generator,
                  distribution: str = 'sekhon_srivastava',
                  show_progessbar: bool = False) -> np.ndarray:
    """
    :param occupancy_ratio:     Ratio of the area of the medium occupied by particles.
    :param precipitation_rate:  Measured in millimeters of equivalent liquid water per hour.
    :param R_0:                 Radius of circular disk that forms the domain of sampling (in meters).
    :param rng:                 Random number generator initialized externally with a random seed.
    :param distribution:        Distribition model of particle diameters.
    :param show_progessbar:     Flag if progressbar should be displayed.

    :return:                    N-by-3 array of sampled particles as disks, where each row contains abscissa and
                                ordinate of disk center and disk radius (in meters).
    """

    if distribution == 'sekhon':
        diameter_rate_parameter = sekhon_srivastava(precipitation_rate)
    elif distribution == 'gunn':
        diameter_rate_parameter = gunn_marshall(precipitation_rate)
    else:
        raise NotImplementedError('Distribution model unknown.')

    diameter_scale_parameter = 1 / diameter_rate_parameter              # in cm

    # Initialize samples to empty set.
    samples = np.zeros((0, 3))

    # Initialize occupied area to 0.
    area_occupied = 0.0

    # Calculate global occupied area across entire domain.
    area_occupied_global = occupancy_ratio * PI * R_0 ** 2

    large_number = 1 / occupancy_ratio
    total = area_occupied_global * large_number + 1

    if show_progessbar:

        pbar = tqdm(total=total, desc='sampling particles',
                    bar_format='{desc}: {percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    else:

        pbar = None

    i = 0
    r_avg = 0

    # Main sampling loop.
    while area_occupied < area_occupied_global:

        # Sample center of particle.
        length = np.sqrt(rng.uniform(0, R_0 ** 2))
        angle = rng.uniform(0, 2) * PI

        x = length * np.cos(angle)
        y = length * np.sin(angle)

        particle_diameter = np.inf
        # Sample diameter of particle from exponential distribution (in millimeters).
        while particle_diameter > 20:   # limit diameter to a maximum of 2cm
            particle_diameter = rng.exponential(diameter_scale_parameter * 10)

        # Convert diameter to meters.
        particle_diameter = particle_diameter / 1000

        # Sample height of particle center relative to examined plane.
        height = rng.uniform(-particle_diameter / 2, particle_diameter / 2)

        # Calculate radius of disk that constitutes the intersection of the sampled ball with the examined plane.
        disk_radius = np.sqrt((particle_diameter / 2) ** 2 - height ** 2)

        # If the disk includes the origin, reject the sample and continue.
        if x ** 2 + y ** 2 <= disk_radius ** 2:
            continue

        # Check whether current particle overlaps with any particle that has already been sampled.
        sample_has_overlap = (samples[:, 0] - x) ** 2 + (samples[:, 1] - y) ** 2 <= (samples[:, 2] + disk_radius) ** 2

        # If yes, reject the sample and continue.
        if np.any(sample_has_overlap):
            continue

        else:

            r_avg = (r_avg * i + disk_radius) / (i+1)
            i += 1

            area = PI * disk_radius ** 2
            area_occupied += area
            samples = np.concatenate((samples, np.array([[x, y, disk_radius]])))

            if pbar:
                pbar.update(area * large_number)
                pbar.set_postfix({'n_sampled': len(samples),
                                  'r_avg': r_avg})

    if pbar:
        pbar.n = total
        pbar.close()

    return samples
# ============================= END VERBATIM =============================
