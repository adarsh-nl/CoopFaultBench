# SPDX-License-Identifier: MIT
# Copied from MultiCorrupt (https://github.com/ika-rwth-aachen/MultiCorrupt), MIT licence, Copyright 2024 Institute for Automotive Engineering of RWTH Aachen University,
#   file converter/lidar.py, function pointsreducing (upstream commit: UNRESOLVED).
# Licensed under the MIT licence in LICENSE in this directory (MultiCorrupt's).
# Changes: none to the function; only this file's imports are new.
"""MultiCorrupt LiDAR points reduction (uniform random point dropout)."""
import numpy as np


""" points missing """
def pointsreducing(pts, severity):
    """
    Simulates missing lidar points based on a given severity level.

    Args:
    pts: A numpy array of lidar points.
    severity: An integer between 1 and 3, where 1 is the least severe and 3 is the most severe.

    Returns:
    A numpy array of lidar points with missing points.
    """
    s = [70, 80, 90][severity - 1]

    size = pts.shape[0]
    nr_of_samps = int(round(size * ((100 - s) / 100)))  # Calculate number of points to keep

    permutations = np.random.permutation(size)  # Generate random permutation of indices
    ind = permutations[:nr_of_samps]  # Select indices for points to keep

    pts = pts[ind]  # Extract points based on selected indices

    return pts
