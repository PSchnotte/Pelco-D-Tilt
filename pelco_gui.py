"""
pelco_gui.py  --  PySide6 GUI for Pelco-D PT-3002
==================================================
Dependencies:  pip install PySide6 pyserial numpy
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
    QFrame, QSplitter,
)

from pelco_calibration import CalibrationTable
from pelco_d_protocol import PT3002, POSITION_TOLERANCE, OVERSHOOT_THRESHOLD


# ---------------------------------------------------------------------------
# Worker  (lives in a QThread; all serial I/O happens here)
# ---------------------------------------------------------------------------

class MoveWorker(QObject):
    position_updated = Signal(float)
    status_message   = Signal(str)
    motion_finished  = Signal()

    def __init__(self, pt: PT3002, stop_event: threading.Event) -> None:
        super().__init__()
        self._pt         = pt
        self._stop_event = stop_event

    def _tilt_to_with_stop(
        self,
        target: float,
        speed: int,
        timeout: float = 40.0,
    ) -> Optional[float]:
        pt = self._pt

        if pt.calibration is not None:
            feedback_tgt = pt.calibration.sensor_to_feedback_target(target)
        else:
            feedback_tgt = target

        current_raw = pt.get_tilt_raw()
        if current_raw is None:
            self.status_message.emit(
                f"[WARNING] pelco-tilt-to: no position feedback"
                f" - aborting move to {target:.1f} deg"
            )
            return None

        if abs(current_raw - feedback_tgt) <= POSITION_TOLERANCE:
            return pt.get_tilt()

        going_up = feedback_tgt > current_raw
        if going_up:
            pt._tilt_increasing(speed)
        else:
            pt._tilt_decreasing(speed)

        poll_interval   = max(0.05, 0.15 - (speed / 63.0) * 0.10)
        brake_margin    = POSITION_TOLERANCE + (speed / 63.0) * 3.0
        t_start         = time.time()
        prev_raw: Optional[float] = current_raw
        wrong_dir_count = 0

        while not self._stop_event.is_set():
            time.sleep(poll_interval)
            pos_raw = pt.get_tilt_raw()
            elapsed = time.time() - t_start

            if pos_raw is not None:
                display = (
                    pt.calibration.feedback_to_sensor(pos_raw)
                    if pt.calibration else pos_raw
                )
                self.position_updated.emit(display)

                if abs(pos_raw - feedback_tgt) <= brake_margin:
                    pt.stop()
                    return pt.get_tilt()

                if prev_raw is not None:
                    moved_wrong = (
                        (going_up     and pos_raw < prev_raw - OVERSHOOT_THRESHOLD) or
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

            if elapsed > timeout:
                pt.stop()
                self.status_message.emit(f"[WARNING] Timeout after {timeout:.0f}s")
                return pt.get_tilt()

        pt.stop()
        return pt.get_tilt()

    def run_goto(self, target: float, speed: int) -> None:
        coord = "sensor-deg" if self._pt.calibration else "pelco-deg"
        self.status_message.emit(f"Moving to {target:.1f} deg ({coord}) ...")
        final = self._tilt_to_with_stop(target, speed)
        if not self._stop_event.is_set():
            self.status_message.emit(
                f"Reached {final:.2f} deg ({coord})."
                if final is not None else "Move failed -- no position feedback."
            )
        else:
            self.status_message.emit("Stopped by user.")
        self.motion_finished.emit()

    def run_pendulum(
        self,
        start: float,
        end: float,
        speed: int,
        passes: int,
    ) -> None:
        coord   = "sensor-deg" if self._pt.calibration else "pelco-deg"
        endless = (passes == 0)
        done    = 0

        self.status_message.emit(
            f"Pendulum: {start:.1f} <-> {end:.1f} deg ({coord})  |  "
            f"speed={speed}  |  passes={'inf' if endless else passes}"
        )

        self.status_message.emit(f"Moving to start {start:.1f} deg ...")
        self._tilt_to_with_stop(start, speed)
        if self._stop_event.is_set():
            self.status_message.emit("Stopped by user.")
            self.motion_finished.emit()
            return

        while not self._stop_event.is_set() and (endless or done < passes):
            done  += 1
            tgt    = end if done % 2 == 1 else start
            label  = "inf" if endless else f"{done}/{passes}"
            self.status_message.emit(f"Pass {label}: -> {tgt:.1f} deg")
            self._tilt_to_with_stop(tgt, speed)
            if not self._stop_event.is_set():
                time.sleep(0.3)

        if self._stop_event.is_set():
            self.status_message.emit("Pendulum stopped by user.")
        else:
            self.status_message.emit(f"Pendulum complete -- {done} passes.")
        self.motion_finished.emit()

    def read_position_once(self) -> None:
        raw = self._pt.get_tilt_raw()
        if raw is not None:
            display = (
                self._pt.calibration.feedback_to_sensor(raw)
                if self._pt.calibration else raw
            )
            self.position_updated.emit(display)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PelcoGUI(QMainWindow):

    _sig_goto     = Signal(float, int)
    _sig_pendulum = Signal(float, float, int, int)
    _sig_poll     = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pelco-D PT-3002 Controller")
        self.setMinimumSize(740, 580)

        self._pt:          Optional[PT3002]           = None
        self._calibration: Optional[CalibrationTable] = None
        self._worker:      Optional[MoveWorker]       = None
        self._thread:      Optional[QThread]          = None
        self._stop_event                              = threading.Event()
        self._motion_active                           = False

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
        root.addWidget(self._build_connection_bar())

        splitter = QSplitter(Qt.Horizontal)
        left     = QWidget()
        ll       = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(10)
        ll.addWidget(self._build_calib_group())
        ll.addWidget(self._build_params_group())
        ll.addWidget(self._build_goto_group())
        ll.addWidget(self._build_pendulum_group())
        ll.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(self._build_log_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
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
        grp = QGroupBox("Calibration CSV")
        lay = QHBoxLayout(grp)
        self._lbl_calib = QLabel("No file loaded -- uncalibrated (raw Pelco deg)")
        self._lbl_calib.setWordWrap(True)
        lay.addWidget(self._lbl_calib, stretch=1)
        btn = QPushButton("Load ...")
        btn.setFixedWidth(68)
        btn.clicked.connect(self._load_calibration)
        lay.addWidget(btn)
        return grp

    def _build_params_group(self) -> QGroupBox:
        grp  = QGroupBox("Motion Parameters")
        grid = QGridLayout(grp)
        grid.setColumnStretch(1, 1)
        grid.addWidget(QLabel("Speed (1-63):"), 0, 0)
        speed_row = QHBoxLayout()
        self._slider_speed = QSlider(Qt.Horizontal)
        self._slider_speed.setRange(1, 63)
        self._slider_speed.setValue(30)
        self._lbl_speed_val = QLabel("30")
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
        self._spin_passes.setValue(4)
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
        grp = QGroupBox("Go to Angle")
        lay = QHBoxLayout(grp)
        lay.addWidget(QLabel("Angle (deg):"))
        self._spin_goto = QDoubleSpinBox()
        self._spin_goto.setRange(-180.0, 180.0)
        self._spin_goto.setDecimals(1)
        self._spin_goto.setValue(0.0)
        self._spin_goto.setSingleStep(1.0)
        self._spin_goto.setFixedWidth(88)
        lay.addWidget(self._spin_goto)
        self._btn_goto = QPushButton("Go to")
        self._btn_goto.setFixedWidth(68)
        self._btn_goto.clicked.connect(self._on_goto)
        lay.addWidget(self._btn_goto)
        lay.addStretch()
        return grp

    def _build_pendulum_group(self) -> QGroupBox:
        grp  = QGroupBox("Pendulum Scan")
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
        lay   = QVBoxLayout(panel)
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
        lay.addWidget(QLabel("Position:"))
        self._lbl_position = QLabel("--")
        self._lbl_position.setFont(QFont("Courier New", 18, QFont.Bold))
        self._lbl_position.setStyleSheet("color:#01696f; min-width:110px;")
        lay.addWidget(self._lbl_position)
        self._lbl_pos_unit = QLabel("(not connected)")
        self._lbl_pos_unit.setStyleSheet("color:#7a7974; font-size:11px;")
        lay.addWidget(self._lbl_pos_unit)
        lay.addStretch()
        self._btn_stop = QPushButton("  STOP")
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
        self._btn_goto.setEnabled(connected and not self._motion_active)
        self._btn_pendulum.setEnabled(connected and not self._motion_active)
        self._btn_stop.setEnabled(connected)
        self._lbl_pos_unit.setText(
            ("sensor-deg (calibrated)" if self._calibration
             else "pelco-deg [uncalibrated]")
            if connected else "(not connected)"
        )
        if connected:
            self._poll_timer.start()
        else:
            self._poll_timer.stop()
            self._lbl_position.setText("--")

    def _set_motion_active(self, active: bool) -> None:
        self._motion_active = active
        self._btn_goto.setEnabled(not active and self._pt is not None)
        self._btn_pendulum.setEnabled(not active and self._pt is not None)
        self._btn_pendulum.setText("Running ..." if active else "Start Pendulum")

    def _log_msg(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        if "[WARNING]" in msg or "failed" in msg.lower():
            line = (f"<span style='color:#7a7974'>{ts}</span>  "
                    f"<span style='color:#964219'>{msg}</span>")
        elif "Stopped" in msg or "STOP" in msg:
            line = (f"<span style='color:#7a7974'>{ts}</span>  "
                    f"<span style='color:#a13544'><b>{msg}</b></span>")
        elif "complete" in msg.lower() or "Reached" in msg:
            line = (f"<span style='color:#7a7974'>{ts}</span>  "
                    f"<span style='color:#437a22'>{msg}</span>")
        else:
            line = f"<span style='color:#7a7974'>{ts}</span>  {msg}"
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
        port    = self._cb_port.currentText()
        baud    = int(self._cb_baud.currentText())
        address = self._spin_addr.value()
        if port in ("(none)", ""):
            self._log_msg("[WARNING] No COM port selected.")
            return
        try:
            self._pt = PT3002(
                port=port, address=address, baudrate=baud,
                calibration=self._calibration,
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
        self._worker.status_message.connect(self._log_msg)
        self._worker.motion_finished.connect(self._on_motion_finished)
        self._sig_goto.connect(self._worker.run_goto)
        self._sig_pendulum.connect(self._worker.run_pendulum)
        self._sig_poll.connect(self._worker.read_position_once)

        self._thread.start()
        self._log_msg(
            f"Connected: <b>{port}</b>  {baud} baud  addr={address}  |  "
            f"calibration: {'loaded' if self._calibration else 'none -- uncalibrated'}"
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration CSV", "",
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            self._calibration = CalibrationTable.from_csv(path)
            short = path.replace("\\", "/").split("/")[-1]
            self._lbl_calib.setText(
                f"{short}  ({len(self._calibration.rows)} pts  |  "
                f"sensor [{self._calibration.sensor_min:.1f} .. "
                f"{self._calibration.sensor_max:.1f} deg])"
            )
            self._log_msg(f"Calibration loaded: <b>{short}</b>")
            if self._pt is not None:
                self._pt.calibration = self._calibration
                self._lbl_pos_unit.setText("sensor-deg (calibrated)")
                self._log_msg("Calibration applied to active connection.")
            self._spin_pend_start.setValue(self._calibration.sensor_min)
            self._spin_pend_end.setValue(self._calibration.sensor_max)
        except Exception as exc:
            self._log_msg(f"[WARNING] Calibration load failed: {exc}")
            self._calibration = None

    # Motion

    def _on_goto(self) -> None:
        if self._pt is None or self._motion_active:
            return
        self._stop_event.clear()
        self._set_motion_active(True)
        self._sig_goto.emit(self._spin_goto.value(), self._slider_speed.value())

    def _on_pendulum(self) -> None:
        if self._pt is None or self._motion_active:
            return
        start  = self._spin_pend_start.value()
        end    = self._spin_pend_end.value()
        if abs(start - end) < 0.1:
            self._log_msg("[WARNING] Start and End angles are identical.")
            return
        passes = 0 if self._chk_infinite.isChecked() else self._spin_passes.value()
        self._stop_event.clear()
        self._set_motion_active(True)
        self._sig_pendulum.emit(start, end, self._slider_speed.value(), passes)

    def _on_stop(self) -> None:
        self._stop_event.set()
        if self._pt:
            try:
                self._pt.stop()
            except Exception:
                pass
        self._log_msg("STOP command sent.")

    def _on_poll_timer(self) -> None:
        if self._motion_active and self._worker is not None:
            self._sig_poll.emit()

    def _on_position_updated(self, pos: float) -> None:
        self._lbl_position.setText(f"{pos:+.2f} deg")

    def _on_motion_finished(self) -> None:
        self._set_motion_active(False)
        if self._pt is not None:
            raw = self._pt.get_tilt_raw()
            if raw is not None:
                display = (
                    self._pt.calibration.feedback_to_sensor(raw)
                    if self._pt.calibration else raw
                )
                self._lbl_position.setText(f"{display:+.2f} deg")

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
    pal.setColor(QPalette.Window,          QColor("#f7f6f2"))
    pal.setColor(QPalette.WindowText,      QColor("#28251d"))
    pal.setColor(QPalette.Base,            QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase,   QColor("#f3f0ec"))
    pal.setColor(QPalette.Text,            QColor("#28251d"))
    pal.setColor(QPalette.Button,          QColor("#edeae5"))
    pal.setColor(QPalette.ButtonText,      QColor("#28251d"))
    pal.setColor(QPalette.Highlight,       QColor("#01696f"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.Disabled, QPalette.Text,       QColor("#bab9b4"))
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