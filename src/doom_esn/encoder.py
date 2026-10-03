"""Frame -> small float vector. Pools to a coarse grid and removes the static background."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import F32


class FrameEncoder:
    def __init__(self, height: int = 24, width: int = 32) -> None:
        self.height, self.width = height, width
        self.mean: F32 = np.zeros(height * width, dtype=np.float32)
        self.scale: float = 1.0

    @property
    def dim(self) -> int:
        return self.height * self.width

    def pool(self, frame: np.ndarray) -> F32:
        """Grey-scale block-average of an (H, W, 3) uint8 frame to (height*width,) in [0, 1]."""
        gray = (
            frame.astype(np.float32).mean(axis=-1) if frame.ndim == 3 else frame.astype(np.float32)
        )
        fh, fw = gray.shape[0] // self.height, gray.shape[1] // self.width
        if fh < 1 or fw < 1:
            raise ValueError(f"frame {gray.shape} smaller than target {self.height}x{self.width}")
        crop = gray[: self.height * fh, : self.width * fw]
        pooled = crop.reshape(self.height, fh, self.width, fw).mean(axis=(1, 3))
        return (pooled.ravel() / 255.0).astype(np.float32)

    def fit(self, pooled: F32) -> FrameEncoder:
        """Fit mean image and one global scale from pooled frames (T, dim)."""
        self.mean = pooled.mean(axis=0).astype(np.float32)
        self.scale = max(float(pooled.std()), 1e-6)
        return self

    def transform_pooled(self, pooled: F32) -> F32:
        return ((pooled - self.mean) / self.scale).astype(np.float32)

    def __call__(self, frame: np.ndarray) -> F32:
        return self.transform_pooled(self.pool(frame))

    def to_arrays(self) -> dict[str, Any]:
        return {"enc_mean": self.mean, "enc_scale": np.float32(self.scale)}

    @classmethod
    def from_arrays(cls, height: int, width: int, arrs: Any) -> FrameEncoder:
        enc = cls(height, width)
        enc.mean = np.asarray(arrs["enc_mean"], dtype=np.float32)
        enc.scale = float(arrs["enc_scale"])
        return enc
