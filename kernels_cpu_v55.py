"""CPU specialized numerical helpers for HUD Fan v5.5."""

from __future__ import annotations

import numpy as np


class FixedDesignLeastSquares:
    """Giải nhiều RHS với cùng ma trận, cùng cutoff hạng số."""

    def __init__(self, design: np.ndarray, rcond: float = 1e-11):
        """Thực thi __init__."""
        matrix = np.asarray(design, dtype=float)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("FIXED_DESIGN_INVALID")
        if not np.isfinite(rcond) or rcond < 0.0:
            raise ValueError("FIXED_DESIGN_RCOND_INVALID")

        self.row_count, self.column_count = matrix.shape
        U, singular, Vh = np.linalg.svd(
            matrix, full_matrices=False
        )
        cutoff = (
            float(rcond) * float(singular[0])
            if len(singular) else 0.0
        )
        keep = singular > cutoff

        self.singular_values = singular.copy()
        self.rank = int(np.count_nonzero(keep))
        self.U = U[:, keep].copy()
        self.s = singular[keep].copy()
        self.Vh = Vh[keep, :].copy()

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Thực thi solve."""
        value = np.asarray(rhs, dtype=float)
        if value.shape != (self.row_count,):
            raise ValueError("FIXED_DESIGN_RHS_SHAPE")
        if not np.all(np.isfinite(value)):
            raise ValueError("FIXED_DESIGN_RHS_NONFINITE")

        if self.rank == 0:
            return np.zeros(self.column_count, dtype=float)

        return self.Vh.T @ ((self.U.T @ value) / self.s)
