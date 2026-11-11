"""
pelco_calibration.py
=====================
Calibration look-up table for the Pelco-D pan/tilt unit.

Each axis (tilt, pan) is calibrated independently using a set of
measured (pelco_deg, measured_deg) pairs, where:
    pelco_deg    -- angle reported/expected by the Pelco unit itself
    measured_deg -- true angle as measured by an external reference sensor

The relationship between pelco_deg and measured_deg is close to linear,
so values requested outside the calibrated range are extrapolated
linearly using the slope of the nearest edge segment, rather than being
clamped to the table boundary. This is intentional: the physical Pelco
unit has no fixed range on pan, and on tilt any mechanical end-stop is
handled by the motion-control layer (overshoot detection in
pelco_d_protocol.py), not by the calibration table.

CSV format (long/combined format, semicolon-separated):
    axis; pelco_deg; measured_deg
    tilt; 2.00; -79.3
    tilt; 10.53; -68.9
    ...
    pan; 334; 100.75
    ...

Dependencies: numpy
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class CalibrationRow:
    """A single measured calibration point for one axis."""
    pelco_deg: float     # angle reported/expected by the Pelco unit
    measured_deg: float  # true angle from the external reference sensor


@dataclass
class CalibrationTable:
    """
    Bidirectional look-up table between Pelco-internal angle and
    externally measured angle, for a single axis (tilt OR pan).

    Values outside the calibrated range are linearly extrapolated
    using the slope of the outermost calibration segment, so the
    table always returns a usable result for any input.
    """
    rows: List[CalibrationRow] = field(default_factory=list)

    # Cached arrays, sorted by measured_deg (ascending)
    _pelco_by_measured: np.ndarray = field(init=False, repr=False)
    _measured_sorted: np.ndarray = field(init=False, repr=False)

    # Cached arrays, sorted by pelco_deg (ascending)
    _measured_by_pelco: np.ndarray = field(init=False, repr=False)
    _pelco_sorted: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.rows:
            self._pelco_by_measured = np.array([])
            self._measured_sorted = np.array([])
            self._measured_by_pelco = np.array([])
            self._pelco_sorted = np.array([])
            return

        # Sort once by measured_deg for measured -> pelco lookups
        by_measured = sorted(self.rows, key=lambda r: r.measured_deg)
        self._measured_sorted = np.array([r.measured_deg for r in by_measured])
        self._pelco_by_measured = np.array([r.pelco_deg for r in by_measured])

        # Sort once by pelco_deg for pelco -> measured lookups
        by_pelco = sorted(self.rows, key=lambda r: r.pelco_deg)
        self._pelco_sorted = np.array([r.pelco_deg for r in by_pelco])
        self._measured_by_pelco = np.array([r.measured_deg for r in by_pelco])

    @property
    def measured_min(self) -> float:
        return float(self._measured_sorted.min()) if len(self._measured_sorted) else 0.0

    @property
    def measured_max(self) -> float:
        return float(self._measured_sorted.max()) if len(self._measured_sorted) else 0.0

    @property
    def pelco_min(self) -> float:
        return float(self._pelco_sorted.min()) if len(self._pelco_sorted) else 0.0

    @property
    def pelco_max(self) -> float:
        return float(self._pelco_sorted.max()) if len(self._pelco_sorted) else 0.0

    @staticmethod
    def _interp_with_linear_extrapolation(
        x: float, xp: np.ndarray, fp: np.ndarray
    ) -> float:
        """
        Like np.interp, but linearly extrapolates beyond the table
        edges instead of clamping to the boundary value. Requires
        xp to be sorted ascending and to contain at least 2 points.
        """
        if len(xp) < 2:
            # Not enough points to extrapolate; fall back to the
            # single known value (or 0.0 if the table is empty).
            return float(fp[0]) if len(fp) else 0.0

        if x < xp[0]:
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            return float(fp[0] + slope * (x - xp[0]))
        if x > xp[-1]:
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            return float(fp[-1] + slope * (x - xp[-1]))
        return float(np.interp(x, xp, fp))

    def measured_to_pelco(self, measured_deg: float) -> float:
        """Desired measured-deg (sensor) -> corresponding Pelco angle."""
        return self._interp_with_linear_extrapolation(
            measured_deg, self._measured_sorted, self._pelco_by_measured
        )

    def pelco_to_measured(self, pelco_deg: float) -> float:
        """Live Pelco angle -> corresponding measured-deg (for display)."""
        return self._interp_with_linear_extrapolation(
            pelco_deg, self._pelco_sorted, self._measured_by_pelco
        )

    @classmethod
    def from_csv(cls, path, axis: str) -> "CalibrationTable":
        """
        Load calibration rows for a single axis from the combined
        long-format CSV (columns: axis; pelco_deg; measured_deg).

        axis: "tilt" or "pan" -- only rows matching this axis are loaded.
        """
        rows: List[CalibrationRow] = []
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh, delimiter=";")
            for line in reader:
                if len(line) < 3:
                    continue
                row_axis = line[0].strip().lower()
                if row_axis != axis.lower():
                    continue
                try:
                    rows.append(CalibrationRow(
                        pelco_deg=float(line[1].strip()),
                        measured_deg=float(line[2].strip()),
                    ))
                except ValueError:
                    # Skip header row or malformed lines
                    continue
        if not rows:
            raise ValueError(f"No valid '{axis}' calibration data in '{path}'")
        return cls(rows=rows)

    def __repr__(self) -> str:
        return (
            f"CalibrationTable({len(self.rows)} pts, "
            f"measured [{self.measured_min:.1f}deg .. {self.measured_max:.1f}deg], "
            f"pelco [{self.pelco_min:.1f}deg .. {self.pelco_max:.1f}deg])"
        )
