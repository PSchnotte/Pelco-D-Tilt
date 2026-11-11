"""
pelco_gui.py -- PySide6 GUI for Pelco-D PT-3002
==================================================
Dependencies: pip install PySide6 pyserial numpy
Place in the same directory as pelco_d_protocol.py and pelco_calibration.py.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import serial
import serial.tools.list_ports

from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider,
    QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QTextEdit,
    QFrame, QSplitter, QScrollArea,
)

from pelco_calibration import CalibrationTable
from pelco_d_protocol import (
    PT3002, POSITION_TOLERANCE, OVERSHOOT_THRESHOLD,
    STALL_TIMEOUT, STALL_EPSILON, MAX_READ_FAILURES,
)

# Text file that receives one line per detected angle change, in the
# form "<timestamp>;<axis>;<angle_deg>". Created next to this script.
ANGLE_LOG_PATH = "pelco_angle_log.txt"

# Minimum angle change (degrees) that triggers a new angle-log line.
# Keeps the log file from being flooded by tiny jitter between polls.
ANGLE_LOG_EPSILON = 0.05

# ---------------------------------------------------------------------------
# Worker (lives in a QThread; all serial I/O happens here)
# ---------------------------------------------------------------------------

class MoveWorker(QObject):
    position_updated = Signal(float)
    pan_position_updated = Signal(float)
    status_message = Signal(str)
    motion_finished = Signal()

    def __init__(self, pt: PT3002, stop_event: threading.Event) -> None:
        super().__init__()
        self._pt = pt
        self._stop_event = stop_event
        # Last logged angle per axis, used to only write a new line
        # when the angle actually changed by more than ANGLE_LOG_EPSILON
        # degrees, instead of on every single poll tick.
        self._last_logged_angle = {"tilt": None, "pan": None}

    def _tilt_to_with_stop(self, target: float, speed: int, timeout: Optional[float] = None):
        pt = self._pt

        # Resolve target (measured/sensor deg if calibrated, else raw
        # Pelco deg) to raw Pelco feedback coordinates. Targets outside
        # the calibrated range are linearly extrapolated by the
        # CalibrationTable rather than rejected -- the unit simply
        # keeps moving until it reaches the extrapolated setpoint or
        # a mechanical end-stop triggers the overshoot detection below.
        if pt.calibration is not None:
            feedback_tgt = pt.calibration.measured_to_pelco(target)
        else:
            feedback_tgt = target

        current_raw = pt.get_tilt_raw()
        if current_raw is None:
            self.status_message.emit(
                f"[WARNING] pelco-tilt-to: no position feedback"
                f" - aborting move to {target:.1f} deg"
            )
            # Try to recover the link automatically so the next command
            # does not fail the same way and require a manual reconnect.
            self._attempt_reset("tilt-to (no initial feedback)")
            return None

        distance = abs(current_raw - feedback_tgt)
        if timeout is None:
            est_deg_per_sec = max(0.3, 1.5 * (speed / 30.0))
            timeout = (distance / est_deg_per_sec) + 15.0

        if distance <= POSITION_TOLERANCE:
            return pt.get_tilt()

        going_up = feedback_tgt > current_raw
        if going_up:
            pt._tilt_increasing(speed)
        else:
            pt._tilt_decreasing(speed)

        poll_interval = max(0.05, 0.15 - (speed / 63.0) * 0.10)
        brake_margin = POSITION_TOLERANCE + (speed / 63.0) * 3.0
        t_start = time.time()
        prev_raw: Optional[float] = current_raw
        wrong_dir_count = 0
        read_failures = 0

        # Stall tracking -- separate from the wrong-direction overshoot
        # check. A stall is "no meaningful position change for
        # STALL_TIMEOUT seconds", which happens when the unit hits a
        # mechanical end-stop and stops responding to the drive command
        # without ever reversing direction.
        stall_ref_raw = current_raw
        stall_ref_time = t_start

        while not self._stop_event.is_set():
            time.sleep(poll_interval)
            pos_raw = pt.get_tilt_raw()
            elapsed = time.time() - t_start

            if pos_raw is None:
                read_failures += 1
                self.status_message.emit(
                    f"[WARNING] Tilt position read failed "
                    f"({read_failures}/{MAX_READ_FAILURES})"
                )
                if read_failures >= MAX_READ_FAILURES:
                    pt.stop()
                    self.status_message.emit(
                        "[WARNING] Tilt feedback lost -- attempting link reset ..."
                    )
                    self._attempt_reset("tilt-to (feedback lost)")
                    return pt.get_tilt()
                if elapsed > timeout:
                    pt.stop()
                    self.status_message.emit(f"[WARNING] Timeout after {timeout:.0f}s")
                    return pt.get_tilt()
                continue

            read_failures = 0
            display = (
                pt.calibration.pelco_to_measured(pos_raw)
                if pt.calibration else pos_raw
            )
            self.position_updated.emit(display)
            self._log_angle_change("tilt", display)

            if abs(pos_raw - feedback_tgt) <= brake_margin:
                pt.stop()
                return pt.get_tilt()

            if prev_raw is not None:
                moved_wrong = (
                    (going_up and pos_raw < prev_raw - OVERSHOOT_THRESHOLD) or
                    (not going_up and pos_raw > prev_raw + OVERSHOOT_THRESHOLD)
                )
                if moved_wrong:
                    wrong_dir_count += 1
                    if wrong_dir_count >= 3:
                        pt.stop()
                        self.status_message.emit(
                            f"[WARNING] Overshoot at raw={pos_raw:.2f} deg"
                        )
                        return pt.get_tilt()
                else:
                    wrong_dir_count = 0
            prev_raw = pos_raw

            # If the position has barely moved in the last STALL_TIMEOUT
            # seconds despite an active drive command, treat it as a
            # mechanical end-stop or blocked motor and stop immediately
            # instead of waiting for the full timeout.
            if abs(pos_raw - stall_ref_raw) > STALL_EPSILON:
                stall_ref_raw = pos_raw
                stall_ref_time = time.time()
            elif time.time() - stall_ref_time > STALL_TIMEOUT:
                pt.stop()
                self.status_message.emit(
                    f"[WARNING] Tilt stalled at raw={pos_raw:.2f} deg "
                    f"(no movement for {STALL_TIMEOUT:.0f}s -- "
                    f"likely mechanical end-stop). Aborting move."
                )
                return pt.get_tilt()

            if elapsed > timeout:
                pt.stop()
                self.status_message.emit(f"[WARNING] Timeout after {timeout:.0f}s")
                return pt.get_tilt()

        pt.stop()
        return pt.get_tilt()

    def _pan_to_with_stop(self, target: float, direction: str, speed: int, timeout: float = 80.0):
        """
        Moves pan continuously in the given direction until pt.get_pan()
        (calibrated measured-deg if pt.pan_calibration is loaded, else
        raw Pelco deg) reports a value within POSITION_TOLERANCE of
        target, then stops.

        Pan rotates continuously (no fixed range), so out-of-table
        targets are handled the same way as tilt: the CalibrationTable
        extrapolates linearly, and the direction is always chosen by
        the caller since the shortest path cannot be inferred.
        """
        pt = self._pt
        current = pt.get_pan()
        if current is None:
            self.status_message.emit("[WARNING] No pan position feedback.")
            self._attempt_reset("pan-to (no initial feedback)")
            return None

        if direction == "left":
            pt._pan_left(speed)
        elif direction == "right":
            pt._pan_right(speed)
        else:
            self.status_message.emit(f"[WARNING] Invalid pan direction: {direction!r}")
            return None

        poll_interval = max(0.05, 0.15 - (speed / 63.0) * 0.10)
        t_start = time.time()
        read_failures = 0

        # Same stall tracking as tilt -- pan has no fixed range, but the
        # physical unit can still jam (cable snag, obstruction), which
        # looks identical to a tilt end-stop from the software side.
        stall_ref_pos = current
        stall_ref_time = t_start

        while not self._stop_event.is_set():
            time.sleep(poll_interval)
            pos = pt.get_pan()
            elapsed = time.time() - t_start

            if pos is None:
                read_failures += 1
                self.status_message.emit(
                    f"[WARNING] Pan position read failed "
                    f"({read_failures}/{MAX_READ_FAILURES})"
                )
                if read_failures >= MAX_READ_FAILURES:
                    pt.stop()
                    self.status_message.emit(
                        "[WARNING] Pan feedback lost -- attempting link reset ..."
                    )
                    self._attempt_reset("pan-to (feedback lost)")
                    return pt.get_pan()
                if elapsed > timeout:
                    pt.stop()
                    self.status_message.emit(f"[WARNING] Pan timeout after {timeout:.0f}s")
                    return pos
                continue

            read_failures = 0
            self.pan_position_updated.emit(pos)
            self._log_angle_change("pan", pos)

            if abs(pos - target) <= POSITION_TOLERANCE:
                pt.stop()
                return pos

            if abs(pos - stall_ref_pos) > STALL_EPSILON:
                stall_ref_pos = pos
                stall_ref_time = time.time()
            elif time.time() - stall_ref_time > STALL_TIMEOUT:
                pt.stop()
                self.status_message.emit(
                    f"[WARNING] Pan stalled at {pos:.2f} deg "
                    f"(no movement for {STALL_TIMEOUT:.0f}s). Aborting move."
                )
                return pos

            if elapsed > timeout:
                pt.stop()
                self.status_message.emit(f"[WARNING] Pan timeout after {timeout:.0f}s")
                return pos

        pt.stop()
        return pt.get_pan()

    def run_goto(self, target: float, speed: int) -> None:
        # The whole move is wrapped in try/except so an unhandled
        # exception (e.g. serial.SerialException from a dropped USB
        # connection) can never silently kill the worker without emitting
        # motion_finished, which would otherwise leave the GUI stuck with
        # every button disabled and no log output.
        coord = "sensor-deg" if self._pt.calibration else "pelco-deg"
        self.status_message.emit(f"Moving to {target:.1f} deg ({coord}) ...")
        try:
            final = self._tilt_to_with_stop(target, speed)
            if not self._stop_event.is_set():
                self.status_message.emit(
                    f"Reached {final:.2f} deg ({coord})."
                    if final is not None else "Move failed -- no position feedback."
                )
            else:
                self.status_message.emit("Stopped by user.")
        except Exception as exc:
            self.status_message.emit(f"[WARNING] Goto failed with exception: {exc}")
            self._attempt_reset("goto (exception)")
        self.motion_finished.emit()

    def run_pan_goto(self, target: float, direction: str, speed: int) -> None:
        """
        Moves pan continuously in the given direction until pt.get_pan()
        reports a value within POSITION_TOLERANCE of target, then stops.
        Uses the pan CalibrationTable (measured-deg) if one is loaded,
        otherwise raw Pelco degrees.
        """
        coord = "sensor-deg" if self._pt.pan_calibration else "pelco-deg"
        self.status_message.emit(
            f"Panning {direction} to {target:.1f} deg ({coord}) ..."
        )
        try:
            final = self._pan_to_with_stop(target, direction, speed)
            if not self._stop_event.is_set():
                self.status_message.emit(
                    f"Pan reached {final:.2f} deg ({coord})."
                    if final is not None else "Pan move failed -- no position feedback."
                )
            else:
                self.status_message.emit("Pan stopped by user.")
        except Exception as exc:
            self.status_message.emit(f"[WARNING] Pan goto failed with exception: {exc}")
            self._attempt_reset("pan-goto (exception)")
        self.motion_finished.emit()

    def run_pan_jog(self, direction: str, duration: float, speed: int) -> None:
        """Time-based open-loop jog -- no position feedback used."""
        pt = self._pt
        self.status_message.emit(f"Pan jog {direction}: {duration:.2f}s ...")
        try:
            pt.pan_jog(direction, duration, speed)
            self.status_message.emit(f"Pan jog {direction} complete.")
        except Exception as exc:
            self.status_message.emit(f"[WARNING] Pan jog failed: {exc}")
            self._attempt_reset("pan-jog (exception)")
        self.motion_finished.emit()

    def run_tilt_jog(self, direction: str, duration: float, speed: int) -> None:
        """Time-based open-loop jog -- no position feedback used."""
        pt = self._pt
        self.status_message.emit(f"Tilt jog {direction}: {duration:.2f}s ...")
        try:
            pt.tilt_jog(direction, duration, speed)
            self.status_message.emit(f"Tilt jog {direction} complete.")
        except Exception as exc:
            self.status_message.emit(f"[WARNING] Tilt jog failed: {exc}")
            self._attempt_reset("tilt-jog (exception)")
        self.motion_finished.emit()

    def run_pendulum(
        self,
        start: float,
        end: float,
        speed: int,
        passes: int,
    ) -> None:
        # The return value of _tilt_to_with_stop() is checked on every
        # pass. A failed/stalled pass (None, or the target simply not
        # reached because of a mechanical end-stop) must not be treated
        # like a successful pass -- consecutive failures abort the
        # pendulum run with a clear log message instead of looping
        # forever and falsely reporting completion.
        coord = "sensor-deg" if self._pt.calibration else "pelco-deg"
        endless = (passes == 0)
        done = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 2

        self.status_message.emit(
            f"Pendulum: {start:.1f} <-> {end:.1f} deg ({coord}) | "
            f"speed={speed} | passes={'inf' if endless else passes}"
        )

        try:
            self.status_message.emit(f"Moving to start {start:.1f} deg ...")
            start_result = self._tilt_to_with_stop(start, speed)
            if self._stop_event.is_set():
                self.status_message.emit("Stopped by user.")
                self.motion_finished.emit()
                return
            if start_result is None:
                self.status_message.emit(
                    "[WARNING] Could not reach pendulum start position -- aborting."
                )
                self.motion_finished.emit()
                return

            while not self._stop_event.is_set() and (endless or done < passes):
                done += 1
                tgt = end if done % 2 == 1 else start
                label = "inf" if endless else f"{done}/{passes}"
                self.status_message.emit(f"Pass {label}: -> {tgt:.1f} deg")
                result = self._tilt_to_with_stop(tgt, speed)

                if self._stop_event.is_set():
                    break

                if result is None:
                    consecutive_failures += 1
                    self.status_message.emit(
                        f"[WARNING] Pass {label} did not complete "
                        f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES} "
                        f"consecutive failures)."
                    )
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        self.status_message.emit(
                            "[WARNING] Pendulum aborted -- repeated failures "
                            "reaching endpoint (check for a mechanical "
                            "end-stop or lost feedback)."
                        )
                        break
                else:
                    consecutive_failures = 0

                if not self._stop_event.is_set():
                    time.sleep(0.3)

            if self._stop_event.is_set():
                self.status_message.emit("Pendulum stopped by user.")
            elif consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                pass  # Already logged above.
            else:
                self.status_message.emit(f"Pendulum complete -- {done} passes.")
        except Exception as exc:
            self.status_message.emit(f"[WARNING] Pendulum failed with exception: {exc}")
            self._attempt_reset("pendulum (exception)")
        self.motion_finished.emit()

    def _attempt_reset(self, context: str) -> None:
        """
        Called whenever a move loop gives up because of lost feedback or
        an unhandled exception. Tries pt.reset_link() (stop + flush
        serial buffers + re-probe position) so the next command from the
        user has a chance to succeed without requiring a manual
        Disconnect/Connect cycle. If the reset probe still fails, this is
        logged clearly so the user knows a real reconnect or hardware
        check is required -- persistent failures are never hidden, only
        one automatic recovery attempt is made.
        """
        self.status_message.emit(f"[WARNING] Attempting automatic link reset ({context}) ...")
        try:
            ok = self._pt.reset_link()
        except Exception as exc:
            ok = False
            self.status_message.emit(f"[WARNING] Reset attempt raised an exception: {exc}")
        if ok:
            self.status_message.emit("Link reset successful -- position feedback restored.")
        else:
            self.status_message.emit(
                "[WARNING] Link reset failed -- unit is still not responding. "
                "A manual Disconnect/Connect (or checking cabling/power) is required."
            )

    def _log_angle_change(self, axis: str, angle_deg: float) -> None:
        """
        Appends "<timestamp>;<axis>;<angle>" to ANGLE_LOG_PATH whenever
        the reported angle changes by more than ANGLE_LOG_EPSILON degrees
        since the last logged value for that axis. Runs in the worker
        thread, so file writes never block the GUI event loop.
        """
        last = self._last_logged_angle.get(axis)
        if last is not None and abs(angle_deg - last) < ANGLE_LOG_EPSILON:
            return
        self._last_logged_angle[axis] = angle_deg
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(ANGLE_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(f"{ts};{axis};{angle_deg:.3f}\n")
        except OSError as exc:
            self.status_message.emit(f"[WARNING] Could not write angle log: {exc}")

    def read_position_once(self) -> None:
        raw = self._pt.get_tilt_raw()
        if raw is not None:
            display = (
                self._pt.calibration.pelco_to_measured(raw)
                if self._pt.calibration else raw
            )
            self.position_updated.emit(display)
            self._log_angle_change("tilt", display)

        pan_raw = self._pt.get_pan_raw()
        if pan_raw is not None:
            pan_display = (
                self._pt.pan_calibration.pelco_to_measured(pan_raw)
                if self._pt.pan_calibration else pan_raw
            )
            self.pan_position_updated.emit(pan_display)
            self._log_angle_change("pan", pan_display)

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PelcoGUI(QMainWindow):

    _sig_goto = Signal(float, int)
    _sig_pendulum = Signal(float, float, int, int)
    _sig_pan_goto = Signal(float, str, int)
    _sig_pan_jog = Signal(str, float, int)
    _sig_tilt_jog = Signal(str, float, int)
    _sig_poll = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pelco-D PT-3002 Controller")
        self.setMinimumSize(860, 620)  # Wide/tall enough for the scrollable left column and log panel

        self._pt: Optional[PT3002] = None
        self._calibration: Optional[CalibrationTable] = None       # tilt
        self._pan_calibration: Optional[CalibrationTable] = None   # pan
        self._worker: Optional[MoveWorker] = None
        self._thread: Optional[QThread] = None
        self._stop_event = threading.Event()
        self._motion_active = False

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(450)
        self._poll_timer.timeout.connect(self._on_poll_timer)

        self._build_ui()
        self._refresh_ports()
        self._set_connected(False)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)

        # -- Left column: starts with Calibration CSV at the top,
        #    aligned with the Connection bar on the right column. --
        # Wrapped in a QScrollArea with a defined minimum width so that,
        # if the splitter or window becomes narrower than the natural
        # width of the stacked group boxes, a vertical scrollbar appears
        # instead of the controls being squeezed and clipped.
        left_content = QWidget()
        ll = QVBoxLayout(left_content)
        ll.setContentsMargins(0, 0, 4, 0)
        ll.setSpacing(10)
        ll.addWidget(self._build_calib_group())
        ll.addWidget(self._build_params_group())
        ll.addWidget(self._build_goto_group())
        ll.addWidget(self._build_jog_group())
        ll.addWidget(self._build_pendulum_group())
        ll.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_content)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.NoFrame)
        # Minimum width wide enough that no child control (fixed-width
        # spin boxes, sliders, labels) ever gets clipped, even if the
        # splitter handle is dragged all the way to its minimum.
        left_scroll.setMinimumWidth(300)
        left_content.setMinimumWidth(280)

        # -- Right column: Connection bar on top, Log panel below. --
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)
        rl.addWidget(self._build_connection_bar())
        rl.addWidget(self._build_log_panel(), stretch=1)
        right.setMinimumWidth(320)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        # Prevents the user from dragging the splitter so far that either
        # side collapses to zero width and becomes inaccessible.
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, stretch=1)
        root.addWidget(self._build_bottom_bar())

    def _build_connection_bar(self) -> QGroupBox:
        grp = QGroupBox("Connection")
        lay = QHBoxLayout(grp)
        lay.addWidget(QLabel("COM Port:"))
        self._cb_port = QComboBox()
        self._cb_port.setMinimumWidth(110)
        lay.addWidget(self._cb_port)
        btn_ref = QPushButton("Refresh")
        btn_ref.setFixedWidth(68)
        btn_ref.clicked.connect(self._refresh_ports)
        lay.addWidget(btn_ref)
        lay.addWidget(QLabel("Baud:"))
        self._cb_baud = QComboBox()
        for b in ["2400", "4800", "9600", "19200", "38400"]:
            self._cb_baud.addItem(b)
        self._cb_baud.setCurrentText("2400")
        self._cb_baud.setFixedWidth(78)
        lay.addWidget(self._cb_baud)
        lay.addWidget(QLabel("Addr:"))
        self._spin_addr = QSpinBox()
        self._spin_addr.setRange(1, 255)
        self._spin_addr.setValue(1)
        self._spin_addr.setFixedWidth(54)
        lay.addWidget(self._spin_addr)
        lay.addStretch()
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.setFixedWidth(90)
        self._btn_connect.clicked.connect(self._toggle_connect)
        lay.addWidget(self._btn_connect)
        return grp

    def _build_calib_group(self) -> QGroupBox:
        # Both axes are loaded from the same combined CSV
        # (columns: axis; pelco_deg; measured_deg), filtered by axis.
        # Kept compact (single-line, elided labels, small font) so it
        # does not dominate the left-hand column.
        grp = QGroupBox("Calibration CSV")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(1, 1)

        small_font = QFont()
        small_font.setPointSize(9)

        btn = QPushButton("Load ...")
        btn.setFixedWidth(68)
        btn.clicked.connect(self._load_calibration)
        grid.addWidget(btn, 0, 0)

        self._lbl_calib_file = QLabel("No file loaded")
        self._lbl_calib_file.setFont(small_font)
        self._lbl_calib_file.setStyleSheet("color:#555;")
        self._lbl_calib_file.setToolTip("No file loaded")
        grid.addWidget(self._lbl_calib_file, 0, 1)

        grid.addWidget(QLabel("Tilt:"), 1, 0)
        self._lbl_calib_tilt = QLabel("uncalibrated (raw Pelco deg)")
        self._lbl_calib_tilt.setFont(small_font)
        self._lbl_calib_tilt.setStyleSheet("color:#555;")
        grid.addWidget(self._lbl_calib_tilt, 1, 1)

        grid.addWidget(QLabel("Pan:"), 2, 0)
        self._lbl_calib_pan = QLabel("uncalibrated (raw Pelco deg)")
        self._lbl_calib_pan.setFont(small_font)
        self._lbl_calib_pan.setStyleSheet("color:#555;")
        grid.addWidget(self._lbl_calib_pan, 2, 1)

        return grp

    def _build_params_group(self) -> QGroupBox:
        grp = QGroupBox("Motion Parameters")
        grid = QGridLayout(grp)
        grid.setColumnStretch(1, 1)
        grid.addWidget(QLabel("Speed (1-63):"), 0, 0)
        speed_row = QHBoxLayout()
        self._slider_speed = QSlider(Qt.Horizontal)
        self._slider_speed.setRange(1, 63)
        self._slider_speed.setValue(5)
        self._lbl_speed_val = QLabel("5")
        self._lbl_speed_val.setFixedWidth(26)
        self._slider_speed.valueChanged.connect(
            lambda v: self._lbl_speed_val.setText(str(v))
        )
        speed_row.addWidget(self._slider_speed)
        speed_row.addWidget(self._lbl_speed_val)
        grid.addLayout(speed_row, 0, 1)
        grid.addWidget(QLabel("Passes:"), 1, 0)
        pass_row = QHBoxLayout()
        self._spin_passes = QSpinBox()
        self._spin_passes.setRange(1, 9999)
        self._spin_passes.setValue(20)
        self._spin_passes.setFixedWidth(64)
        self._chk_infinite = QCheckBox("Infinite")
        self._chk_infinite.toggled.connect(
            lambda c: self._spin_passes.setEnabled(not c)
        )
        pass_row.addWidget(self._spin_passes)
        pass_row.addWidget(self._chk_infinite)
        pass_row.addStretch()
        grid.addLayout(pass_row, 1, 1)
        return grp

    def _build_goto_group(self) -> QGroupBox:
        # Pure "fly to an absolute angle" controls -- no jog controls here.
        grp = QGroupBox("Go to Angle")
        outer = QVBoxLayout(grp)

        # -- Tilt row --
        tilt_row = QHBoxLayout()
        tilt_row.addWidget(QLabel("Tilt (deg):"))
        self._spin_goto = QDoubleSpinBox()
        self._spin_goto.setRange(-180.0, 180.0)
        self._spin_goto.setDecimals(1)
        self._spin_goto.setValue(0.0)
        self._spin_goto.setSingleStep(1.0)
        self._spin_goto.setFixedWidth(88)
        tilt_row.addWidget(self._spin_goto)
        self._btn_goto = QPushButton("Go to")
        self._btn_goto.setFixedWidth(68)
        self._btn_goto.clicked.connect(self._on_goto)
        tilt_row.addWidget(self._btn_goto)
        tilt_row.addStretch()
        outer.addLayout(tilt_row)

        # -- Pan row --
        pan_row = QHBoxLayout()
        pan_row.addWidget(QLabel("Pan (deg):"))
        self._spin_pan_goto = QDoubleSpinBox()
        self._spin_pan_goto.setRange(-100000.0, 100000.0)  # no fixed range
        self._spin_pan_goto.setDecimals(1)
        self._spin_pan_goto.setValue(0.0)
        self._spin_pan_goto.setSingleStep(1.0)
        self._spin_pan_goto.setFixedWidth(88)
        pan_row.addWidget(self._spin_pan_goto)

        self._cb_pan_dir = QComboBox()
        self._cb_pan_dir.addItems(["left", "right"])
        self._cb_pan_dir.setFixedWidth(70)
        pan_row.addWidget(self._cb_pan_dir)

        self._btn_pan_goto = QPushButton("Go to")
        self._btn_pan_goto.setFixedWidth(68)
        self._btn_pan_goto.clicked.connect(self._on_pan_goto)
        pan_row.addWidget(self._btn_pan_goto)
        pan_row.addStretch()
        outer.addLayout(pan_row)

        return grp

    def _build_jog_group(self) -> QGroupBox:
        # Open-loop, time-based nudging for both axes -- separated from
        # "Go to Angle" since jogging does not target an absolute angle.
        grp = QGroupBox("Jog")
        outer = QVBoxLayout(grp)

        duration_row = QHBoxLayout()
        duration_row.addWidget(QLabel("Jog duration (s):"))
        self._spin_jog_duration = QDoubleSpinBox()
        self._spin_jog_duration.setRange(0.05, 10.0)
        self._spin_jog_duration.setDecimals(2)
        self._spin_jog_duration.setValue(0.2)
        self._spin_jog_duration.setSingleStep(0.05)
        self._spin_jog_duration.setFixedWidth(80)
        duration_row.addWidget(self._spin_jog_duration)
        duration_row.addStretch()
        outer.addLayout(duration_row)

        # -- Tilt jog row --
        tilt_jog_row = QHBoxLayout()
        tilt_jog_row.addWidget(QLabel("Tilt:"))
        self._btn_tilt_jog_up = QPushButton("\u25b2")   # up arrow
        self._btn_tilt_jog_up.setFixedWidth(36)
        self._btn_tilt_jog_up.clicked.connect(lambda: self._on_tilt_jog("up"))
        tilt_jog_row.addWidget(self._btn_tilt_jog_up)

        self._btn_tilt_jog_down = QPushButton("\u25bc")  # down arrow
        self._btn_tilt_jog_down.setFixedWidth(36)
        self._btn_tilt_jog_down.clicked.connect(lambda: self._on_tilt_jog("down"))
        tilt_jog_row.addWidget(self._btn_tilt_jog_down)
        tilt_jog_row.addStretch()
        outer.addLayout(tilt_jog_row)

        # -- Pan jog row --
        pan_jog_row = QHBoxLayout()
        pan_jog_row.addWidget(QLabel("Pan:"))
        self._btn_pan_jog_left = QPushButton("\u25C0")
        self._btn_pan_jog_left.setFixedWidth(36)
        self._btn_pan_jog_left.clicked.connect(lambda: self._on_pan_jog("left"))
        pan_jog_row.addWidget(self._btn_pan_jog_left)

        self._btn_pan_jog_right = QPushButton("\u25b6")
        self._btn_pan_jog_right.setFixedWidth(36)
        self._btn_pan_jog_right.clicked.connect(lambda: self._on_pan_jog("right"))
        pan_jog_row.addWidget(self._btn_pan_jog_right)
        pan_jog_row.addStretch()
        outer.addLayout(pan_jog_row)

        return grp

    def _build_pendulum_group(self) -> QGroupBox:
        grp = QGroupBox("Pendulum Scan")
        grid = QGridLayout(grp)
        grid.setColumnStretch(1, 1)
        grid.addWidget(QLabel("Start (deg):"), 0, 0)
        self._spin_pend_start = QDoubleSpinBox()
        self._spin_pend_start.setRange(-180.0, 180.0)
        self._spin_pend_start.setDecimals(1)
        self._spin_pend_start.setValue(-31.4)
        self._spin_pend_start.setSingleStep(1.0)
        grid.addWidget(self._spin_pend_start, 0, 1)
        grid.addWidget(QLabel("End (deg):"), 1, 0)
        self._spin_pend_end = QDoubleSpinBox()
        self._spin_pend_end.setRange(-180.0, 180.0)
        self._spin_pend_end.setDecimals(1)
        self._spin_pend_end.setValue(0.9)
        self._spin_pend_end.setSingleStep(1.0)
        grid.addWidget(self._spin_pend_end, 1, 1)
        self._btn_pendulum = QPushButton("Start Pendulum")
        self._btn_pendulum.clicked.connect(self._on_pendulum)
        grid.addWidget(self._btn_pendulum, 2, 0, 1, 2)
        return grp

    def _build_log_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        hdr = QLabel("Status Log")
        hdr.setStyleSheet("font-weight:600; color:#444;")
        lay.addWidget(hdr)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        self._log.setStyleSheet(
            "background:#fafaf8; border:1px solid #d4d1ca; border-radius:6px;"
        )
        lay.addWidget(self._log)
        return panel

    def _build_bottom_bar(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame{background:#f3f0ec;border:1px solid #d4d1ca;border-radius:8px;}"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.addWidget(QLabel("Tilt:"))
        self._lbl_position = QLabel("--")
        self._lbl_position.setFont(QFont("Courier New", 18, QFont.Bold))
        self._lbl_position.setStyleSheet("color:#01696f; min-width:110px;")
        lay.addWidget(self._lbl_position)

        lay.addWidget(QLabel("Pan:"))
        self._lbl_pan_position = QLabel("--")
        self._lbl_pan_position.setFont(QFont("Courier New", 18, QFont.Bold))
        self._lbl_pan_position.setStyleSheet("color:#01696f; min-width:110px;")
        lay.addWidget(self._lbl_pan_position)

        self._lbl_pos_unit = QLabel("(not connected)")
        self._lbl_pos_unit.setStyleSheet("color:#7a7974; font-size:11px;")
        lay.addWidget(self._lbl_pos_unit)
        lay.addStretch()
        self._btn_stop = QPushButton(" STOP")
        self._btn_stop.setFixedSize(110, 44)
        self._btn_stop.setStyleSheet(
            "QPushButton{background:#a13544;color:white;font-size:14px;"
            "font-weight:700;border-radius:8px;border:none;}"
            "QPushButton:hover{background:#782b33;}"
            "QPushButton:pressed{background:#521f24;}"
            "QPushButton:disabled{background:#c8c6c3;color:#888;}"
        )
        self._btn_stop.clicked.connect(self._on_stop)
        lay.addWidget(self._btn_stop)
        return frame

    # UI state

    def _set_connected(self, connected: bool) -> None:
        self._btn_connect.setText("Disconnect" if connected else "Connect")
        self._btn_connect.setStyleSheet(
            "QPushButton{background:#964219;color:white;border-radius:6px;"
            "padding:4px 10px;border:none;}"
            "QPushButton:hover{background:#713417;}"
            if connected else ""
        )
        self._btn_tilt_jog_up.setEnabled(connected and not self._motion_active)
        self._btn_tilt_jog_down.setEnabled(connected and not self._motion_active)
        self._btn_pan_jog_left.setEnabled(connected and not self._motion_active)
        self._btn_pan_jog_right.setEnabled(connected and not self._motion_active)

        self._btn_goto.setEnabled(connected and not self._motion_active)
        self._btn_pan_goto.setEnabled(connected and not self._motion_active)
        self._btn_pendulum.setEnabled(connected and not self._motion_active)
        self._btn_stop.setEnabled(connected)

        if connected:
            tilt_txt = "sensor-deg" if self._calibration else "pelco-deg [uncalibrated]"
            pan_txt = "sensor-deg" if self._pan_calibration else "pelco-deg [uncalibrated]"
            self._lbl_pos_unit.setText(f"Tilt: {tilt_txt} | Pan: {pan_txt}")
        else:
            self._lbl_pos_unit.setText("(not connected)")

        if connected:
            self._poll_timer.start()
        else:
            self._poll_timer.stop()
            self._lbl_position.setText("--")
            self._lbl_pan_position.setText("--")

    def _set_motion_active(self, active: bool) -> None:
        self._motion_active = active
        self._btn_goto.setEnabled(not active and self._pt is not None)
        self._btn_pan_goto.setEnabled(not active and self._pt is not None)
        self._btn_pendulum.setEnabled(not active and self._pt is not None)
        self._btn_pendulum.setText("Running ..." if active else "Start Pendulum")

        self._btn_tilt_jog_up.setEnabled(not active and self._pt is not None)
        self._btn_tilt_jog_down.setEnabled(not active and self._pt is not None)
        self._btn_pan_jog_left.setEnabled(not active and self._pt is not None)
        self._btn_pan_jog_right.setEnabled(not active and self._pt is not None)

    def _log_msg(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"{ts} {msg}"
        self._log.append(line)

    # Connection

    def _refresh_ports(self) -> None:
        current = self._cb_port.currentText()
        self._cb_port.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if not ports:
            self._cb_port.addItem("(none)")
        else:
            for p in ports:
                self._cb_port.addItem(p)
            if current in ports:
                self._cb_port.setCurrentText(current)

    def _toggle_connect(self) -> None:
        if self._pt is not None:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        port = self._cb_port.currentText()
        baud = int(self._cb_baud.currentText())
        address = self._spin_addr.value()
        if port in ("(none)", ""):
            self._log_msg("[WARNING] No COM port selected.")
            return
        try:
            self._pt = PT3002(
                port=port, address=address, baudrate=baud,
                calibration=self._calibration,
                pan_calibration=self._pan_calibration,
            )
        except serial.SerialException as exc:
            self._log_msg(f"[WARNING] Connection failed: {exc}")
            self._pt = None
            return

        self._stop_event.clear()
        self._thread = QThread(self)
        self._worker = MoveWorker(self._pt, self._stop_event)
        self._worker.moveToThread(self._thread)

        self._worker.position_updated.connect(self._on_position_updated)
        self._worker.pan_position_updated.connect(self._on_pan_position_updated)
        self._worker.status_message.connect(self._log_msg)
        self._worker.motion_finished.connect(self._on_motion_finished)
        self._sig_goto.connect(self._worker.run_goto)
        self._sig_pendulum.connect(self._worker.run_pendulum)
        self._sig_pan_goto.connect(self._worker.run_pan_goto)
        self._sig_pan_jog.connect(self._worker.run_pan_jog)
        self._sig_tilt_jog.connect(self._worker.run_tilt_jog)
        self._sig_poll.connect(self._worker.read_position_once)

        self._thread.start()
        self._log_msg(
            f"Connected: {port} {baud} baud addr={address} | "
            f"tilt calibration: {'loaded' if self._calibration else 'none -- uncalibrated'} | "
            f"pan calibration: {'loaded' if self._pan_calibration else 'none -- uncalibrated'}"
        )
        self._set_connected(True)

    def _disconnect(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        if self._pt:
            try:
                self._pt.close()
            except Exception:
                pass
        self._pt = None
        self._worker = None
        self._thread = None
        self._stop_event.clear()
        self._set_connected(False)
        self._set_motion_active(False)
        self._log_msg("Disconnected.")

    # Calibration

    def _load_calibration(self) -> None:
        """
        Loads both tilt and pan calibration tables from a single
        combined CSV (columns: axis; pelco_deg; measured_deg).
        Each axis is filtered independently, so a file missing one
        of the two axes still loads the other successfully.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration CSV", "",
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return

        short = path.replace("\\", "/").split("/")[-1]
        loaded_any = False

        try:
            self._calibration = CalibrationTable.from_csv(path, axis="tilt")
            self._lbl_calib_tilt.setText(
                f"{len(self._calibration.rows)} pts | "
                f"[{self._calibration.measured_min:.1f}..{self._calibration.measured_max:.1f} deg]"
            )
            self._spin_pend_start.setValue(self._calibration.measured_min)
            self._spin_pend_end.setValue(self._calibration.measured_max)
            loaded_any = True
        except ValueError as exc:
            self._calibration = None
            self._lbl_calib_tilt.setText("uncalibrated (raw Pelco deg)")
            self._log_msg(f"[WARNING] Tilt calibration not loaded: {exc}")

        try:
            self._pan_calibration = CalibrationTable.from_csv(path, axis="pan")
            self._lbl_calib_pan.setText(
                f"{len(self._pan_calibration.rows)} pts | "
                f"[{self._pan_calibration.measured_min:.1f}..{self._pan_calibration.measured_max:.1f} deg]"
            )
            loaded_any = True
        except ValueError as exc:
            self._pan_calibration = None
            self._lbl_calib_pan.setText("uncalibrated (raw Pelco deg)")
            self._log_msg(f"[WARNING] Pan calibration not loaded: {exc}")

        if loaded_any:
            self._lbl_calib_file.setText(short)
            self._lbl_calib_file.setToolTip(path)
            self._log_msg(f"Calibration file loaded: {short}")
        else:
            self._lbl_calib_file.setText("No file loaded")
            self._lbl_calib_file.setToolTip("No file loaded")

        if self._pt is not None:
            self._pt.calibration = self._calibration
            self._pt.pan_calibration = self._pan_calibration
            self._log_msg("Calibration applied to active connection.")
            self._set_connected(True)

    # Motion

    def _on_goto(self) -> None:
        if self._pt is None or self._motion_active:
            return
        self._stop_event.clear()
        self._set_motion_active(True)
        self._sig_goto.emit(self._spin_goto.value(), self._slider_speed.value())

    def _on_pan_goto(self) -> None:
        if self._pt is None or self._motion_active:
            return
        self._stop_event.clear()
        self._set_motion_active(True)
        self._sig_pan_goto.emit(
            self._spin_pan_goto.value(),
            self._cb_pan_dir.currentText(),
            self._slider_speed.value(),
        )

    def _on_pan_jog(self, direction: str) -> None:
        if self._pt is None or self._motion_active:
            return
        self._stop_event.clear()
        self._set_motion_active(True)
        self._sig_pan_jog.emit(
            direction,
            self._spin_jog_duration.value(),
            self._slider_speed.value(),
        )

    def _on_tilt_jog(self, direction: str) -> None:
        if self._pt is None or self._motion_active:
            return
        self._stop_event.clear()
        self._set_motion_active(True)
        self._sig_tilt_jog.emit(
            direction,
            self._spin_jog_duration.value(),
            self._slider_speed.value(),
        )

    def _on_pendulum(self) -> None:
        if self._pt is None or self._motion_active:
            return
        start = self._spin_pend_start.value()
        end = self._spin_pend_end.value()
        if abs(start - end) < 0.1:
            self._log_msg("[WARNING] Start and End angles are identical.")
            return
        passes = 0 if self._chk_infinite.isChecked() else self._spin_passes.value()
        self._stop_event.clear()
        self._set_motion_active(True)
        self._sig_pendulum.emit(start, end, self._slider_speed.value(), passes)

    def _on_stop(self) -> None:
        self._stop_event.set()
        self._log_msg("STOP requested -- waiting for worker to halt ...")

    def _on_poll_timer(self) -> None:
        if self._motion_active and self._worker is not None:
            self._sig_poll.emit()

    def _on_position_updated(self, pos: float) -> None:
        self._lbl_position.setText(f"{pos:+.2f} deg")

    def _on_pan_position_updated(self, pos: float) -> None:
        self._lbl_pan_position.setText(f"{pos:+.2f} deg")

    def _on_motion_finished(self) -> None:
        self._set_motion_active(False)
        if self._pt is not None:
            raw = self._pt.get_tilt_raw()
            if raw is not None:
                display = (
                    self._pt.calibration.pelco_to_measured(raw)
                    if self._pt.calibration else raw
                )
                self._lbl_position.setText(f"{display:+.2f} deg")

            pan_raw = self._pt.get_pan_raw()
            if pan_raw is not None:
                pan_display = (
                    self._pt.pan_calibration.pelco_to_measured(pan_raw)
                    if self._pt.pan_calibration else pan_raw
                )
                self._lbl_pan_position.setText(f"{pan_display:+.2f} deg")

    def closeEvent(self, event) -> None:
        self._disconnect()
        event.accept()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f7f6f2"))
    pal.setColor(QPalette.WindowText, QColor("#28251d"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f3f0ec"))
    pal.setColor(QPalette.Text, QColor("#28251d"))
    pal.setColor(QPalette.Button, QColor("#edeae5"))
    pal.setColor(QPalette.ButtonText, QColor("#28251d"))
    pal.setColor(QPalette.Highlight, QColor("#01696f"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#bab9b4"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#bab9b4"))
    app.setPalette(pal)

    app.setStyleSheet(
        "QGroupBox{font-weight:600;border:1px solid #d4d1ca;border-radius:8px;"
        "margin-top:10px;padding:12px 8px 8px 8px;}"
        "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;}"
        "QPushButton{border-radius:6px;padding:5px 14px;"
        "border:1px solid #d4d1ca;background:#edeae5;}"
        "QPushButton:hover{background:#e0ddd8;}"
        "QPushButton:pressed{background:#d4d1ca;}"
        "QPushButton:disabled{background:#f3f0ec;color:#bab9b4;}"
        "QComboBox,QSpinBox,QDoubleSpinBox{"
        "border:1px solid #d4d1ca;border-radius:5px;"
        "padding:3px 6px;background:#ffffff;}"
        "QSlider::groove:horizontal{height:4px;background:#d4d1ca;border-radius:2px;}"
        "QSlider::handle:horizontal{background:#01696f;border-radius:7px;"
        "width:14px;height:14px;margin:-5px 0;}"
        "QSlider::sub-page:horizontal{background:#01696f;border-radius:2px;}"
    )

    win = PelcoGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()