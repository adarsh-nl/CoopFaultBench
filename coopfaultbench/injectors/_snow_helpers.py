# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Numerical helpers used by the LiDAR snow simulation: a RANSAC polynomial
fit and an exact-match binary search. Written for CoopFaultBench from a
behavioural specification."""

import numpy as np

NOT_FOUND = -1


def ransac_polynomial_fit(x, y, order=3, n=15, k=100, t=0.1, d=15, f=0.8):
    """Fit a polynomial to (x, y) robustly with random sample consensus.

    The initial best model is a least-squares fit over all points, scored by
    the summed absolute residual over all points. Then, k times, n indices
    are drawn (with replacement) from numpy's legacy global generator, a
    candidate is fitted to them, and the points whose absolute residual is
    below t form the inlier set. If the inlier count exceeds both d and
    len(x) * f, the inliers are refitted; the refit replaces the best model
    when its summed absolute residual over the inliers is strictly lower.

    Args:
        x: 1-D float array of abscissae.
        y: 1-D float array of ordinates, same length as x.
        order: polynomial degree.
        n: sample size per iteration.
        k: number of iterations (always all performed).
        t: inlier threshold on the absolute residual (strict).
        d: minimum inlier count (strict).
        f: minimum inlier fraction of all points (strict).

    Returns:
        Coefficient array in numpy.polyfit order (highest power first)."""
    num_points = len(x)

    best_model = np.polyfit(x, y, order)
    best_error = np.sum(np.abs(np.polyval(best_model, x) - y))

    for _ in range(k):
        picks = np.random.randint(len(x), size=n)
        candidate = np.polyfit(x[picks], y[picks], order)

        inliers = np.abs(np.polyval(candidate, x) - y) < t
        inlier_count = int(np.count_nonzero(inliers))

        if inlier_count > d and inlier_count > num_points * f:
            x_in = x[inliers]
            y_in = y[inliers]
            refined = np.polyfit(x_in, y_in, order)
            refined_error = np.sum(np.abs(np.polyval(refined, x_in) - y_in))
            if refined_error < best_error:
                best_model = refined
                best_error = refined_error

    return best_model


def find_sorted_index(values, low, high, target):
    """Locate target in an ascending sequence by binary search.

    Searches the inclusive index range low..high and returns an index i with
    values[i] == target, or NOT_FOUND (-1) once the range is empty. Equality
    is tested first, then "greater than"; any other outcome (smaller, or
    unordered such as NaN) continues in the upper half. An out-of-range probe
    raises IndexError, which is not caught.

    Args:
        values: ascending sequence of floats.
        low: first index of the search range (inclusive).
        high: last index of the search range (inclusive).
        target: value to find.

    Returns:
        The matching index, or NOT_FOUND if no element equals target."""
    while low <= high:
        centre = (low + high) // 2
        probe = values[centre]
        if probe == target:
            return centre
        if probe > target:
            high = centre - 1
            continue
        low = centre + 1
    return NOT_FOUND
