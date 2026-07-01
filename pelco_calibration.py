"""
pelco_calibration.py
====================
Calibration look-up table built from measured CSV data.

CSV column order (header optional):
    angle_in,  angle_pelco,  angle_sensor

Dependencies:  numpy
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class CalibrationRow:
    angle_in:     float   # commanded angle  [deg]
    angle_pelco:  float   # Pelco internal feedback [deg]
    angle_sensor: float   # external sensor          [deg]


@dataclass
class CalibrationTable:
    rows: List[CalibrationRow] = field(default_factory=list)

    _cmd:    np.ndarray = field(init=False, repr=False)
    _pelco:  np.ndarray = field(init=False, repr=False)
    _sensor: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.rows:
            self._cmd = self._pelco = self._sensor = np.array([])
            return
        self.rows.sort(key=lambda r: r.angle_sensor)
        self._cmd    = np.array([r.angle_in     for r in self.rows])
        self._pelco  = np.array([r.angle_pelco  for r in self.rows])
        self._sensor = np.array([r.angle_sensor for r in self.rows])

    @property
    def sensor_min(self) -> float:
        return float(self._sensor.min()) if len(self._sensor) else 0.0

    @property
    def sensor_max(self) -> float:
        return float(self._sensor.max()) if len(self._sensor) else 0.0

    def sensor_to_command(self, sensor_deg: float) -> float:
        """Desired sensor-deg -> Pelco command-deg."""
        return float(np.interp(sensor_deg, self._sensor, self._cmd))

    def sensor_to_feedback_target(self, sensor_deg: float) -> float:
        """Desired sensor-deg -> expected Pelco feedback value (brake setpoint)."""
        return float(np.interp(sensor_deg, self._sensor, self._pelco))

    def feedback_to_sensor(self, pelco_feedback: float) -> float:
        """Live Pelco feedback-deg -> sensor-deg (for display)."""
        idx = np.argsort(self._pelco)
        return float(np.interp(pelco_feedback, self._pelco[idx], self._sensor[idx]))

    @classmethod
    def from_csv(cls, path) -> "CalibrationTable":
        rows: List[CalibrationRow] = []
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for line in csv.reader(fh):
                if len(line) < 3:
                    continue
                try:
                    rows.append(CalibrationRow(
                        angle_in     = float(line[0].strip()),
                        angle_pelco  = float(line[1].strip()),
                        angle_sensor = float(line[2].strip()),
                    ))
                except ValueError:
                    continue
        if not rows:
            raise ValueError(f"No valid calibration data in '{path}'")
        return cls(rows=rows)

    def __repr__(self) -> str:
        return (
            f"CalibrationTable({len(self.rows)} pts, "
            f"sensor [{self.sensor_min:.1f}deg .. {self.sensor_max:.1f}deg])"
        )