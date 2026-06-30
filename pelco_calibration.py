# flake8: noqa
"""
pelco_calibration.py
====================
Angle calibration for the Pelco-D PT-3002 pan-tilt unit.

Coordinate systems
------------------
  Sensor  : 0° = horizontal, positive values = above horizon,
             negative values = below horizon.
             This is the physically meaningful reference frame (IMU / inclinometer).
  Pelco command (angle_in)    : The angle value sent to the controller.
             2° = mechanical upper hard-stop (near-vertical),  ~66° = horizontal.
  Pelco feedback (angle_delta): The angle the internal encoder reports back
             when angle_in was commanded.  Systematically ~0.5-0.75° higher than
             angle_in due to the P-controller dead-band.  At the hard-stop (0°
             command) the controller floors at 2°.

Calibration strategy
--------------------
The CSV table stores all three measured columns: angle_in, angle_delta,
angle_sensor.

Two separate interpolation maps are built:
  1. sensor -> angle_in   : Used to compute the correct command to send.
  2. sensor -> angle_delta: Used to compute the expected feedback value at the
                            target position.  tilt_to() compares get_tilt_raw()
                            against this value, not against angle_in.  This
                            eliminates the ~0.5-0.75° systematic stop error.
  3. angle_delta -> sensor: Reverse map, used to convert get_tilt_raw() readings
                            back into physically meaningful sensor degrees for the
                            console display and return value of get_tilt().

If no calibration file is found the class falls back to raw Pelco-degree mode
(legacy) and emits a warning.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in fallback calibration data
# Measured 2026-06-30, PT-3002, COM4 / 2400 baud / address 1
# Tuple layout: (sensor_deg, angle_in_deg, angle_delta_deg)
# The row with angle_in=0° is the mechanical hard-stop; the controller reports
# 2° as the minimum feedback value.
# ---------------------------------------------------------------------------
_DEFAULT_TABLE: list[tuple[float, float, float]] = [
    (-79.3,  0.0,  2.00),   # mechanical hard-stop
    (-68.9, 10.0, 10.53),
    (-56.6, 20.0, 20.52),
    (-43.9, 30.0, 30.68),
    (-31.4, 40.0, 40.53),
    (-18.7, 50.0, 50.75),
    ( -6.3, 60.0, 60.69),
    (  0.9, 66.0, 66.68),   # approx. horizontal
]

# CSV column names (header row is required in the file)
_COL_SENSOR = "angle_sensor"
_COL_IN     = "angle_in"
_COL_DELTA  = "angle_delta"


@dataclass
class CalibrationTable:
    """
    Tri-column calibration table: sensor degrees, Pelco command, Pelco feedback.

    Parameters
    ----------
    rows   : List of (sensor_deg, angle_in_deg, angle_delta_deg) tuples.
             Must be sorted by sensor_deg in strictly ascending order.
    source : Human-readable string describing the data origin (path or 'built-in').
    """

    rows: list[tuple[float, float, float]] = field(
        default_factory=lambda: list(_DEFAULT_TABLE)
    )
    source: str = "built-in default"

    def __post_init__(self) -> None:
        self._sensor_arr = np.array([r[0] for r in self.rows], dtype=float)
        self._in_arr     = np.array([r[1] for r in self.rows], dtype=float)
        self._delta_arr  = np.array([r[2] for r in self.rows], dtype=float)

        if not np.all(np.diff(self._sensor_arr) > 0):
            raise ValueError(
                "Calibration rows must be sorted by sensor_deg in strictly ascending order."
            )
        if not np.all(np.diff(self._in_arr) > 0):
            raise ValueError(
                "angle_in values must be strictly ascending (monotonic mapping required)."
            )
        if not np.all(np.diff(self._delta_arr) > 0):
            raise ValueError(
                "angle_delta values must be strictly ascending (monotonic mapping required)."
            )

        self.sensor_min: float       = float(self._sensor_arr[0])
        self.sensor_max: float       = float(self._sensor_arr[-1])
        self.pelco_in_min: float     = float(self._in_arr[0])
        self.pelco_in_max: float     = float(self._in_arr[-1])
        self.pelco_delta_min: float  = float(self._delta_arr[0])
        self.pelco_delta_max: float  = float(self._delta_arr[-1])

        logger.info(
            "CalibrationTable loaded from '%s': %d points | "
            "sensor [%.1f°..%.1f°]  angle_in [%.1f°..%.1f°]  "
            "angle_delta [%.2f°..%.2f°]",
            self.source, len(self.rows),
            self.sensor_min,      self.sensor_max,
            self.pelco_in_min,    self.pelco_in_max,
            self.pelco_delta_min, self.pelco_delta_max,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def sensor_to_command(self, sensor_deg: float) -> float:
        """
        Convert a target sensor angle to the Pelco command angle (angle_in).

        This is the value written into the RS-485 motion command.

        Parameters
        ----------
        sensor_deg : Target angle in sensor coordinates (degrees).

        Returns
        -------
        float : Pelco command angle (degrees).
        """
        clamped = self._clamp_sensor(sensor_deg)
        return float(np.interp(clamped, self._sensor_arr, self._in_arr))

    def sensor_to_feedback_target(self, sensor_deg: float) -> float:
        """
        Convert a target sensor angle to the expected Pelco feedback value
        (angle_delta) that the encoder will report when the target is reached.

        Use this as the stop criterion in the motion-control loop to fully
        compensate the internal P-controller offset.

        Parameters
        ----------
        sensor_deg : Target angle in sensor coordinates (degrees).

        Returns
        -------
        float : Expected Pelco feedback angle (degrees) at the target position.
        """
        clamped = self._clamp_sensor(sensor_deg)
        return float(np.interp(clamped, self._sensor_arr, self._delta_arr))

    def feedback_to_sensor(self, delta_deg: float) -> float:
        """
        Convert a raw Pelco feedback reading (angle_delta) back to sensor degrees.

        Used by get_tilt() to convert get_tilt_raw() into a physically
        meaningful angle for display and return values.

        Parameters
        ----------
        delta_deg : Pelco encoder feedback angle (degrees).

        Returns
        -------
        float : Corresponding sensor angle (degrees).
        """
        clamped = float(np.clip(delta_deg, self.pelco_delta_min, self.pelco_delta_max))
        if clamped != delta_deg:
            logger.warning(
                "feedback_to_sensor: %.2f° outside calibrated range "
                "[%.2f°..%.2f°] — clamped.",
                delta_deg, self.pelco_delta_min, self.pelco_delta_max,
            )
        return float(np.interp(clamped, self._delta_arr, self._sensor_arr))

    # -----------------------------------------------------------------------
    # Factory helpers
    # -----------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str) -> "CalibrationTable":
        """
        Load a calibration table from a semicolon-separated CSV file.

        Required columns (header row mandatory, spaces around names are ignored):
            angle_in ; angle_delta ; angle_sensor

        Additional columns are silently ignored, so the original measurement
        file format is accepted without modification.

        Parameters
        ----------
        path : Absolute or relative path to the CSV file.

        Returns
        -------
        CalibrationTable

        Raises
        ------
        FileNotFoundError : File does not exist.
        KeyError          : A required column is missing.
        ValueError        : Fewer than 2 data rows, or non-numeric value found.
        ValueError        : Table is not monotonic after sorting.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Calibration file not found: '{path}'")

        rows: list[tuple[float, float, float]] = []
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            if reader.fieldnames is None:
                raise ValueError(f"Calibration file '{path}' appears to be empty.")
            # Strip surrounding whitespace from all column names
            reader.fieldnames = [n.strip() for n in reader.fieldnames]

            for lineno, raw_row in enumerate(reader, start=2):
                row = {k: v.strip() for k, v in raw_row.items()}
                try:
                    sensor_val = float(row[_COL_SENSOR])
                    in_val     = float(row[_COL_IN])
                    delta_val  = float(row[_COL_DELTA])
                except KeyError as exc:
                    raise KeyError(
                        f"Required column {exc} not found in '{path}'. "
                        f"Available columns: {list(row.keys())}"
                    ) from exc
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value on line {lineno} of '{path}': {exc}"
                    ) from exc
                rows.append((sensor_val, in_val, delta_val))

        if len(rows) < 2:
            raise ValueError(
                f"Calibration file '{path}' must contain at least 2 data rows."
            )

        # Sort ascending by sensor angle (allows rows in any order in the file)
        rows.sort(key=lambda r: r[0])
        return cls(rows=rows, source=os.path.abspath(path))

    @classmethod
    def from_default(cls) -> "CalibrationTable":
        """Return a CalibrationTable using the built-in default measurements."""
        return cls(rows=list(_DEFAULT_TABLE), source="built-in default")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _clamp_sensor(self, sensor_deg: float) -> float:
        """Clamp sensor_deg to the calibrated range and log a warning if needed."""
        clamped = float(np.clip(sensor_deg, self.sensor_min, self.sensor_max))
        if clamped != sensor_deg:
            logger.warning(
                "Requested sensor angle %.2f° is outside calibrated range "
                "[%.1f°..%.1f°] — clamped to %.2f°.",
                sensor_deg, self.sensor_min, self.sensor_max, clamped,
            )
        return clamped

    def __repr__(self) -> str:
        return (
            f"CalibrationTable(points={len(self.rows)}, source='{self.source}', "
            f"sensor=[{self.sensor_min:.1f}°..{self.sensor_max:.1f}°], "
            f"pelco_in=[{self.pelco_in_min:.1f}°..{self.pelco_in_max:.1f}°], "
            f"pelco_delta=[{self.pelco_delta_min:.2f}°..{self.pelco_delta_max:.2f}°])"
        )