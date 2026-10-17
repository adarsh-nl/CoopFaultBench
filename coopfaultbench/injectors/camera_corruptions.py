# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Camera corruptions (MultiCorrupt functions via the Apache-2.0 _vendor/imagecorruptions backend).

Requires the optional extra `camera` (opencv-python-headless, scikit-image).
"""
from __future__ import annotations
import numpy as np
from ._vendor.imagecorruptions import image_corruptions as _ic

class BrightnessInjector:
    def __init__(self, severity: int = 2, seed: int = 1000):
        if severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
        self.severity, self.seed = severity, seed

    def __call__(self, image: np.ndarray) -> np.ndarray:
        np.random.seed(self.seed)                      # MultiCorrupt seeds numpy globally
        out = np.asarray(_ic.brightness(np.asarray(image), self.severity))
        return np.clip(out, 0, 255).astype(np.uint8)


class DarknessInjector:
    def __init__(self, severity: int = 2, seed: int = 1000):
        if severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
        self.severity, self.seed = severity, seed

    def __call__(self, image: np.ndarray) -> np.ndarray:
        np.random.seed(self.seed)                      # MultiCorrupt seeds numpy globally
        out = np.asarray(_ic.darkness(np.asarray(image), self.severity))
        return np.clip(out, 0, 255).astype(np.uint8)


class CameraFogInjector:
    def __init__(self, severity: int = 2, seed: int = 1000):
        if severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
        self.severity, self.seed = severity, seed

    def __call__(self, image: np.ndarray) -> np.ndarray:
        np.random.seed(self.seed)                      # MultiCorrupt seeds numpy globally
        out = np.asarray(_ic.fog(np.asarray(image), self.severity))
        return np.clip(out, 0, 255).astype(np.uint8)


class CameraSnowInjector:
    def __init__(self, severity: int = 2, seed: int = 1000):
        if severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
        self.severity, self.seed = severity, seed

    def __call__(self, image: np.ndarray) -> np.ndarray:
        np.random.seed(self.seed)                      # MultiCorrupt seeds numpy globally
        out = np.asarray(_ic.snow(np.asarray(image), self.severity))
        return np.clip(out, 0, 255).astype(np.uint8)
