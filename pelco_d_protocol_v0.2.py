# flake8: noqa
"""
pelco_d_protocol.py
===================
Pelco-D PT-3002 — Sky Scan Script
----------------------------------
Tilts a line camera across the sky between two configurable angle limits.
Supports single-pass and pendulum (back-and-forth) scan patterns with a
configurable number of passes or infinite loop.

Angle convention (when calibration is active)
---------------------------------------------
All user-facing angles — TILT_START, TILT_END, TILT_HOME, tilt_to() — are in
SENSOR degrees:
    0°  = horizontal
   +x°  = above horizon (positive elevation)
   -x°  = below horizon (negative elevation)

The calibration table handles the full conversion chain:
  sensor_deg  ->  angle_in       (Pelco command sent over RS-485)
  sensor_deg  ->  angle_delta    (expected feedback; used as stop criterion)
  angle_delta ->  sensor_deg     (convert get_tilt_raw() for display)

Without a calibration file the script operates in raw Pelco-degree mode
(legacy) and prints a warning.  All TILT_* constants must then be expressed
in raw Pelco degrees.

Confirmed hardware settings:
    Port=COM4, Baudrate=2400, Address=1

Dependencies:
    pip install pyserial numpy
"""

from __future__ import annotations

import logging
import sys
import time
from enum import IntFlag
from typing import Optional

import serial

from pelco_calibration import CalibrationTable

# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pelco")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PORT      = "COM4"
BAUDRATE  = 2400
ADDRESS   = 1
CMD_DELAY = 0.05   # seconds enforced between consecutive write commands

# ------------------------------------------------------------------------------
# Tilt angle limits.
# When calibration is active: sensor degrees (0° = horizontal).
# When running uncalibrated : raw Pelco degrees.
# ------------------------------------------------------------------------------
TILT_START = -31.4   # calibrated: corresponds to ~40° Pelco command
TILT_END   =   0.9   # calibrated: corresponds to ~66° Pelco command (~horizontal)
TILT_HOME  =  -0.3   # calibrated: corresponds to ~65° Pelco command

# Movement speed: 0-63
# 6  ~= 1 deg/s    16 ~= 2.5 deg/s    32 ~= 5 deg/s    48 ~= 7.5 deg/s    63 = max
SCAN_SPEED = 30

# "single"   — one pass TILT_START -> TILT_END, then return and stop
# "pendulum" — back-and-forth: START->END->START->END ...
SCAN_PATTERN = "pendulum"

# Number of passes (0 = infinite loop, stop with Ctrl+C)
PASS_COUNT = 4

# Seconds to pause at each endpoint before the next move
ENDPOINT_PAUSE = 0.5

# Tilt direction inversion flag.
# On this device: increasing angle_in requires sending TILT_DOWN.
TILT_INVERTED = True

# Stop criterion tolerance (in Pelco feedback degrees).
# The motion loop compares get_tilt_raw() against the calibrated feedback target.
POSITION_TOLERANCE = 0.5   # degrees

# Overshoot guard: only trigger reversal detection after this much wrong-direction
# movement, to suppress mechanical vibration / encoder noise.
OVERSHOOT_THRESHOLD = 3.0  # degrees (Pelco feedback degrees)

# ------------------------------------------------------------------------------
# Calibration file path — semicolon-separated CSV (see pelco_calibration.py).
# Set to None to run uncalibrated (Pelco-degree mode, legacy).
# ------------------------------------------------------------------------------
CALIBRATION_FILE: Optional[str] = "pelco_angle_calibration.csv"

# ==============================================================================
# PELCO-D PROTOCOL LAYER
# ==============================================================================

class Cmd2(IntFlag):
    TILT_DOWN = 0x10
    TILT_UP   = 0x08
    PAN_LEFT  = 0x04
    PAN_RIGHT = 0x02


class Opcode:
    QUERY_TILT_POS = 0x53
    QUERY_PAN_POS  = 0x51


def _build_frame(addr: int, cmd1: int, cmd2: int, d1: int, d2: int) -> bytes:
    """Assemble a 7-byte Pelco-D command frame with checksum."""
    chk = (addr + cmd1 + cmd2 + d1 + d2) % 256
    return bytes([0xFF, addr, cmd1, cmd2, d1, d2, chk])


def _build_ext(addr: int, opcode: int, d1: int, d2: int) -> bytes:
    """Assemble a Pelco-D extended (query) frame."""
    return _build_frame(addr, 0x00, opcode, d1, d2)


# ==============================================================================
# PT3002 CONTROLLER
# ==============================================================================

class PT3002:
    """
    Driver for the Pelco-D PT-3002 pan-tilt unit.

    Public interface (tilt_to, get_tilt, get_pan) operates in the active
    coordinate system:
      - Sensor degrees when a CalibrationTable is attached.
      - Raw Pelco degrees when running uncalibrated (legacy mode).

    The internal motion loop compares raw Pelco feedback values against the
    calibration-derived feedback target (angle_delta), which eliminates the
    ~0.5-0.75° systematic P-controller offset present in the raw angle_in target.
    """

    MAX_SPEED = 0x3F  # 63

    def __init__(
        self,
        port: str        = PORT,
        address: int     = ADDRESS,
        baudrate: int    = BAUDRATE,
        cmd_delay: float = CMD_DELAY,
        calibration: Optional[CalibrationTable] = None,
    ) -> None:
        self.addr        = address
        self.delay       = cmd_delay
        self.calibration = calibration

        if calibration is not None:
            logger.info("Calibration active: %r", calibration)
        else:
            logger.warning(
                "No calibration table — operating in raw Pelco-degree mode. "
                "TILT_* constants must be in Pelco degrees. "
                "Angles will NOT correspond to physical sensor readings."
            )

        self.ser = serial.Serial(
            port      = port,
            baudrate  = baudrate,
            bytesize  = serial.EIGHTBITS,
            parity    = serial.PARITY_NONE,
            stopbits  = serial.STOPBITS_ONE,
            timeout   = 0.5,
            dsrdtr    = False,
            rtscts    = False,
        )
        self.ser.dtr = False
        self.ser.rts = False
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        logger.info("Connected: %s @ %d baud, address %d", port, baudrate, address)

    # --------------------------------------------------------------------------
    # Context manager
    # --------------------------------------------------------------------------

    def __enter__(self) -> "PT3002":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self.ser.is_open:
            self.stop()
            self.ser.close()

    # --------------------------------------------------------------------------
    # Serial I/O primitives
    # --------------------------------------------------------------------------

    def _send(self, frame: bytes) -> None:
        """Write a Pelco-D frame and enforce the inter-command delay."""
        self.ser.write(frame)
        time.sleep(self.delay)

    def _query(self, frame: bytes, expected: int = 7) -> Optional[bytes]:
        """
        Send a query frame and read the response.

        At 2400 baud a 7-byte response takes ~29 ms.
        A 60 ms wait provides 2x margin without wasting polling budget.
        """
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        time.sleep(0.06)
        data = self.ser.read(self.ser.in_waiting or expected)
        return data if len(data) >= expected else None

    # --------------------------------------------------------------------------
    # Low-level motion commands (always in raw Pelco degrees)
    # --------------------------------------------------------------------------

    def _tilt_increasing(self, speed: int) -> None:
        """Send the command that causes the Pelco internal angle to increase."""
        cmd = Cmd2.TILT_DOWN if TILT_INVERTED else Cmd2.TILT_UP
        self._send(_build_frame(self.addr, 0, int(cmd), 0, min(speed, self.MAX_SPEED)))

    def _tilt_decreasing(self, speed: int) -> None:
        """Send the command that causes the Pelco internal angle to decrease."""
        cmd = Cmd2.TILT_UP if TILT_INVERTED else Cmd2.TILT_DOWN
        self._send(_build_frame(self.addr, 0, int(cmd), 0, min(speed, self.MAX_SPEED)))

    def stop(self) -> None:
        """Send an all-zero stop command."""
        self._send(_build_frame(self.addr, 0, 0, 0, 0))

    # --------------------------------------------------------------------------
    # Position queries
    # --------------------------------------------------------------------------

    def get_tilt_raw(self) -> Optional[float]:
        """
        Read the raw Pelco-internal tilt feedback angle (degrees).

        The PT-3002 encodes negative angles as (raw - 360°). The 180° threshold
        distinguishes negative from positive per the Pelco-D spec for this unit.

        Returns None on communication timeout.
        """
        resp = self._query(_build_ext(self.addr, Opcode.QUERY_TILT_POS, 0, 0))
        if resp and len(resp) >= 7:
            raw = (resp[4] << 8) | resp[5]
            return (raw - 36000) / 100.0 if raw > 18000 else raw / 100.0
        return None

    def get_pan_raw(self) -> Optional[float]:
        """
        Read the raw Pelco-internal pan angle (degrees).

        Returns None on communication timeout.
        """
        resp = self._query(_build_ext(self.addr, Opcode.QUERY_PAN_POS, 0, 0))
        if resp and len(resp) >= 7:
            return ((resp[4] << 8) | resp[5]) / 100.0
        return None

    def get_tilt(self) -> Optional[float]:
        """
        Read current tilt in the active coordinate system.

        Returns sensor degrees when calibration is active, raw Pelco degrees
        otherwise. Returns None on communication timeout.
        """
        raw = self.get_tilt_raw()
        if raw is None:
            return None
        if self.calibration is not None:
            return self.calibration.feedback_to_sensor(raw)
        return raw

    def get_pan(self) -> Optional[float]:
        """
        Read current pan angle (raw Pelco degrees; no calibration applied).

        Returns None on communication timeout.
        """
        return self.get_pan_raw()

    # --------------------------------------------------------------------------
    # High-level move
    # --------------------------------------------------------------------------

    def tilt_to(
        self,
        target_deg: float,
        speed: int     = SCAN_SPEED,
        timeout: float = 40.0,
    ) -> Optional[float]:
        """
        Move to target_deg (active coordinate system) and block until the
        position is reached or the timeout expires.

        Motion-control algorithm
        ------------------------
        1. Convert target_deg to:
             a. pelco_cmd    — angle_in value written into the RS-485 command.
             b. feedback_tgt — angle_delta value expected from get_tilt_raw()
                               when the target is reached.
           In uncalibrated mode both equal target_deg.
        2. Send a continuous-motion command in the correct direction.
        3. Poll get_tilt_raw() at a rate that scales with speed.
        4. Apply a speed-dependent brake margin to stop before mechanical
           overshoot:  brake_margin = POSITION_TOLERANCE + (speed / 63) * 3.0°
        5. Guard against encoder noise / inertia with a consecutive wrong-
           direction counter (OVERSHOOT_THRESHOLD, 3 consecutive samples).

        Using feedback_tgt (angle_delta) as the stop criterion instead of
        pelco_cmd (angle_in) compensates the ~0.5-0.75° P-controller offset,
        resulting in a more accurate final position.

        Parameters
        ----------
        target_deg : Target angle in the active coordinate system.
        speed      : Pelco speed value 0-63.
        timeout    : Maximum wait time in seconds before aborting.

        Returns
        -------
        float  : Final position in the active coordinate system.
        None   : If repeated communication failure prevented movement.
        """
        # --- Coordinate conversion ---
        if self.calibration is not None:
            pelco_cmd    = self.calibration.sensor_to_command(target_deg)
            feedback_tgt = self.calibration.sensor_to_feedback_target(target_deg)
            logger.debug(
                "tilt_to: sensor=%.2f° -> cmd=%.2f°  feedback_tgt=%.2f°",
                target_deg, pelco_cmd, feedback_tgt,
            )
        else:
            pelco_cmd    = target_deg
            feedback_tgt = target_deg

        # --- Read current raw feedback position ---
        current_raw = self.get_tilt_raw()
        if current_raw is None:
            logger.warning(
                "tilt_to: no position feedback — aborting move to %.2f°.", target_deg
            )
            return None

        if abs(current_raw - feedback_tgt) <= POSITION_TOLERANCE:
            logger.info(
                "tilt_to: already within tolerance (raw=%.2f°, tgt=%.2f°).",
                current_raw, feedback_tgt,
            )
            return self.get_tilt()

        # --- Start continuous motion toward the feedback target ---
        going_up = feedback_tgt > current_raw
        if going_up:
            self._tilt_increasing(speed)
        else:
            self._tilt_decreasing(speed)

        # Poll interval: shorter at higher speed to catch the brake point in time
        poll_interval = max(0.05, 0.15 - (speed / 63.0) * 0.10)

        # Brake margin: stop earlier at high speed to compensate for inertia
        brake_margin = POSITION_TOLERANCE + (speed / 63.0) * 3.0

        t_start                    = time.time()
        prev_raw: Optional[float]  = current_raw
        wrong_dir_count            = 0

        while True:
            time.sleep(poll_interval)
            pos_raw = self.get_tilt_raw()
            elapsed = time.time() - t_start

            if pos_raw is not None:
                remaining = abs(pos_raw - feedback_tgt)

                pos_display = (
                    self.calibration.feedback_to_sensor(pos_raw)
                    if self.calibration else pos_raw
                )
                print(
                    f"  Tilt: {pos_display:7.2f}°  ->  target: {target_deg:.2f}°  "
                    f"(delta_feedback {remaining:.2f}°)    ",
                    end="\r",
                )

                # --- Brake point reached ---
                if remaining <= brake_margin:
                    self.stop()
                    final = self.get_tilt()
                    final_display = final if final is not None else pos_display
                    print(
                        f"  Tilt: {final_display:7.2f}°  ->  target: {target_deg:.2f}°"
                        f"  [stopped]                  "
                    )
                    return final

                # --- Overshoot guard ---
                if prev_raw is not None:
                    moved_wrong = (
                        (going_up     and pos_raw < prev_raw - OVERSHOOT_THRESHOLD) or
                        (not going_up and pos_raw > prev_raw + OVERSHOOT_THRESHOLD)
                    )
                    if moved_wrong:
                        wrong_dir_count += 1
                        if wrong_dir_count >= 3:
                            self.stop()
                            logger.warning(
                                "tilt_to: consistent overshoot at raw=%.2f° "
                                "(feedback_tgt=%.2f°).",
                                pos_raw, feedback_tgt,
                            )
                            return self.get_tilt()
                    else:
                        wrong_dir_count = 0

                prev_raw = pos_raw

            # --- Timeout ---
            if elapsed > timeout:
                self.stop()
                logger.warning(
                    "tilt_to: timeout after %.1f s (last raw=%.2f°, tgt=%.2f°).",
                    timeout,
                    pos_raw if pos_raw is not None else float("nan"),
                    feedback_tgt,
                )
                return self.get_tilt()


# ==============================================================================
# SKY SCAN ROUTINE
# ==============================================================================

def run_sky_scan(pt: PT3002) -> None:
    """
    Execute the sky scan according to the global SCAN_* configuration constants.

    All angle arguments to tilt_to() are in the active coordinate system.
    """
    endless = (PASS_COUNT == 0)
    pattern = SCAN_PATTERN.lower()
    done    = 0

    print(f"\n  Moving to home position ({TILT_HOME:.1f}°) ...")
    pt.tilt_to(TILT_HOME, speed=SCAN_SPEED)
    time.sleep(ENDPOINT_PAUSE)

    print(f"\n  Moving to start position ({TILT_START:.1f}°) ...")
    pt.tilt_to(TILT_START, speed=SCAN_SPEED)
    time.sleep(ENDPOINT_PAUSE)

    try:
        while endless or done < PASS_COUNT:
            done  += 1
            label  = "inf" if endless else f"{done}/{PASS_COUNT}"

            if pattern == "single":
                print(f"\n  Pass {label}: {TILT_START:.1f}° -> {TILT_END:.1f}°")
                pt.tilt_to(TILT_END, speed=SCAN_SPEED)
                time.sleep(ENDPOINT_PAUSE)
                if not endless and done >= PASS_COUNT:
                    break
                print("  Returning to start ...")
                pt.tilt_to(TILT_START, speed=SCAN_SPEED)
                time.sleep(ENDPOINT_PAUSE)

            elif pattern == "pendulum":
                a = TILT_START if done % 2 == 1 else TILT_END
                b = TILT_END   if done % 2 == 1 else TILT_START
                print(f"\n  Pass {label}: {a:.1f}° -> {b:.1f}°")
                pt.tilt_to(b, speed=SCAN_SPEED)
                time.sleep(ENDPOINT_PAUSE)

            else:
                raise ValueError(f"Unknown SCAN_PATTERN: {pattern!r}")

    except KeyboardInterrupt:
        print("\n\n  Scan stopped by user (Ctrl+C).")
        pt.tilt_to(TILT_HOME, speed=SCAN_SPEED)

    else:
        print("\n  Scan complete — returning to home position ...")
        pt.tilt_to(TILT_HOME, speed=SCAN_SPEED)

    pt.stop()
    final_tilt = pt.get_tilt()
    final_pan  = pt.get_pan()
    coord_lbl  = "sensor deg" if pt.calibration else "pelco deg"
    print(
        f"\n  Final position — Pan: {final_pan}° (pelco)   "
        f"Tilt: {final_tilt}° ({coord_lbl})"
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def _load_calibration() -> Optional[CalibrationTable]:
    """
    Attempt to load the calibration table from CALIBRATION_FILE.

    Returns the CalibrationTable on success.
    Returns None if the file is absent (triggers legacy/uncalibrated mode).
    Raises on malformed data so errors are never silently swallowed.
    """
    if not CALIBRATION_FILE:
        logger.warning("CALIBRATION_FILE is not configured — running uncalibrated.")
        return None

    try:
        return CalibrationTable.from_csv(CALIBRATION_FILE)
    except FileNotFoundError:
        logger.warning(
            "Calibration file '%s' not found — "
            "running in uncalibrated (raw Pelco-degree) mode.",
            CALIBRATION_FILE,
        )
        return None


def main() -> None:
    calib = _load_calibration()
    coord = (
        "sensor degrees  (0°=horizontal, negative=below horizon)"
        if calib else
        "raw Pelco degrees  [UNCALIBRATED — no calibration file]"
    )

    print("=" * 62)
    print("  Pelco-D PT-3002 — Sky Scan")
    print("=" * 62)
    print(f"  Port      : {PORT}   Baud={BAUDRATE}   Address={ADDRESS}")
    print(f"  Tilt range: {TILT_START:.1f}° -> {TILT_END:.1f}°")
    print(f"  Home      : {TILT_HOME:.1f}°")
    print(f"  Angle unit: {coord}")
    print(f"  Speed     : {SCAN_SPEED}/63")
    print(f"  Pattern   : {SCAN_PATTERN}")
    print(f"  Passes    : {PASS_COUNT if PASS_COUNT > 0 else 'inf (Ctrl+C to stop)'}")
    print(f"  Inverted  : {TILT_INVERTED}")
    print(f"  Calibration: {calib or 'NONE'}")
    print()

    try:
        pt = PT3002(calibration=calib)
    except serial.SerialException as exc:
        logger.error("Could not open serial port: %s", exc)
        sys.exit(1)

    with pt:
        run_sky_scan(pt)


if __name__ == "__main__":
    main()