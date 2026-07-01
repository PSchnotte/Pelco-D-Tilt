# flake8: noqa
"""
Pelco-D PT-3002 — Controller + Sky Scan Script
===============================================
Tilts a line camera across the sky between two configurable
angle limits. Supports single-pass and pendulum (back-and-forth)
scan patterns, with a configurable number of passes or infinite loop.

Confirmed hardware settings:
Port=COM4, Baudrate=2400, Address=1

Dependencies:
pip install pyserial
"""

import serial
import time
from enum import IntFlag
from typing import Optional

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

PORT     = "COM4"
BAUDRATE = 2400
ADDRESS  = 1
CMD_DELAY = 0.05

# Tilt angle limits (degrees)
TILT_START = 40.0
TILT_END   = 90.0

# Movement speed: 0-63
SCAN_SPEED = 30

# "single" or "pendulum"
SCAN_PATTERN = "pendulum"

# Number of passes (0 = infinite loop)
PASS_COUNT = 4

# Seconds to pause at each endpoint before reversing
ENDPOINT_PAUSE = 0.5

# FIX 1: Tilt direction is inverted on this device.
TILT_INVERTED = True

# FIX 2: Overshoot tolerance
OVERSHOOT_THRESHOLD = 3.0  # degrees

# How close to target before stopping (degrees)
POSITION_TOLERANCE = 0.8

# Tilt angle for home/park position
TILT_HOME = 65.0

# ══════════════════════════════════════════════════════════════

# ── Pelco-D Protocol ──────────────────────────────────────────

class Cmd2(IntFlag):
    TILT_DOWN = 0x10
    TILT_UP   = 0x08
    PAN_LEFT  = 0x04
    PAN_RIGHT = 0x02

class Opcode:
    QUERY_TILT_POS = 0x53
    QUERY_PAN_POS  = 0x51

def _build_frame(addr, cmd1, cmd2, d1, d2):
    chk = (addr + cmd1 + cmd2 + d1 + d2) % 256
    return bytes([0xFF, addr, cmd1, cmd2, d1, d2, chk])

def _build_ext(addr, opcode, d1, d2):
    return _build_frame(addr, 0x00, opcode, d1, d2)


# ── PT3002 Controller ─────────────────────────────────────────

class PT3002:
    MAX_SPEED = 0x3F

    def __init__(
        self,
        port       = PORT,
        address    = ADDRESS,
        baudrate   = BAUDRATE,
        cmd_delay  = CMD_DELAY,
        calibration = None,      # CalibrationTable | None
    ):
        self.addr        = address
        self.delay       = cmd_delay
        self.calibration = calibration   # used by GUI / MoveWorker

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
        print(f"  Connected: {port} @ {baudrate} baud, address {address}")

    def close(self):
        if self.ser.is_open:
            self.stop()
            self.ser.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def _send(self, frame):
        self.ser.write(frame)
        time.sleep(self.delay)

    def _query(self, frame, n=7):
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        time.sleep(0.3)          # wait for controller response
        data = self.ser.read(self.ser.in_waiting or n)
        return data if len(data) >= n else None

    def stop(self):
        self._send(_build_frame(self.addr, 0, 0, 0, 0))

    def _tilt_increasing(self, speed):
        """Send the command that makes tilt angle increase."""
        cmd = Cmd2.TILT_DOWN if TILT_INVERTED else Cmd2.TILT_UP
        self._send(_build_frame(self.addr, 0, int(cmd),
                                0, min(speed, self.MAX_SPEED)))

    def _tilt_decreasing(self, speed):
        """Send the command that makes tilt angle decrease."""
        cmd = Cmd2.TILT_UP if TILT_INVERTED else Cmd2.TILT_DOWN
        self._send(_build_frame(self.addr, 0, int(cmd),
                                0, min(speed, self.MAX_SPEED)))

    def get_tilt_raw(self) -> Optional[float]:
        """
        Returns the raw Pelco internal feedback angle in degrees,
        without any calibration correction. Used by the motion loop
        as the brake setpoint.  Returns None on timeout.
        """
        resp = self._query(_build_ext(self.addr, Opcode.QUERY_TILT_POS, 0, 0))
        if resp and len(resp) >= 7:
            raw = (resp[4] << 8) | resp[5]
            return (raw - 36000) / 100.0 if raw > 18000 else raw / 100.0
        return None

    def get_tilt(self) -> Optional[float]:
        """
        Returns current tilt in degrees.
        If a CalibrationTable is loaded, returns the calibrated
        sensor-coordinate value.  Otherwise returns raw Pelco degrees.
        """
        raw = self.get_tilt_raw()
        if raw is None:
            return None
        if self.calibration is not None:
            return self.calibration.feedback_to_sensor(raw)
        return raw

    def get_pan(self) -> Optional[float]:
        """Returns current pan in degrees, or None on timeout."""
        resp = self._query(_build_ext(self.addr, Opcode.QUERY_PAN_POS, 0, 0))
        if resp and len(resp) >= 7:
            return ((resp[4] << 8) | resp[5]) / 100.0
        return None

    def tilt_to(self, target_deg, speed=SCAN_SPEED, timeout=40.0):
        """
        Moves tilt continuously toward target_deg and stops
        when position feedback is within POSITION_TOLERANCE degrees.
        NOTE: This blocking version is used by the standalone script only.
              The GUI uses MoveWorker._tilt_to_with_stop() instead.
        """
        current = self.get_tilt_raw()
        if current is None:
            print(f"  [WARNING] pelco-tilt-to: no position feedback"
                  f" - aborting move to {target_deg:.1f} deg")
            return None

        # Resolve target to raw Pelco feedback coordinates
        if self.calibration is not None:
            feedback_tgt = self.calibration.sensor_to_feedback_target(target_deg)
        else:
            feedback_tgt = target_deg

        if abs(current - feedback_tgt) <= POSITION_TOLERANCE:
            return self.get_tilt()

        going_up = feedback_tgt > current
        if going_up:
            self._tilt_increasing(speed)
        else:
            self._tilt_decreasing(speed)

        poll_interval  = max(0.05, 0.15 - (speed / 63.0) * 0.10)
        t_start        = time.time()
        prev_pos       = current
        wrong_dir_count = 0

        while True:
            time.sleep(poll_interval)
            pos     = self.get_tilt_raw()
            elapsed = time.time() - t_start

            if pos is not None:
                remaining    = abs(pos - feedback_tgt)
                brake_margin = POSITION_TOLERANCE + (speed / 63.0) * 3.0
                print(f"  Tilt: {pos:6.2f} deg -> target: {target_deg:.2f} deg"
                      f"  (delta {remaining:.2f} deg) ", end="\r")

                if remaining <= brake_margin:
                    self.stop()
                    print(f"  Tilt: {pos:6.2f} deg -> target: {target_deg:.2f} deg")
                    return self.get_tilt()

                if prev_pos is not None:
                    moved_wrong = (
                        (going_up     and pos < prev_pos - OVERSHOOT_THRESHOLD) or
                        (not going_up and pos > prev_pos + OVERSHOOT_THRESHOLD)
                    )
                    if moved_wrong:
                        wrong_dir_count += 1
                        if wrong_dir_count >= 3:
                            self.stop()
                            print(f"\n  [WARNING] Overshoot at {pos:.2f} deg")
                            return self.get_tilt()
                    else:
                        wrong_dir_count = 0
                prev_pos = pos

            if elapsed > timeout:
                self.stop()
                print(f"\n  [WARNING] Timeout after {timeout}s at {pos} deg")
                return self.get_tilt()


# ── Sky Scan (standalone script) ──────────────────────────────

def run_sky_scan(pt: PT3002):
    endless = (PASS_COUNT == 0)
    pattern = SCAN_PATTERN.lower()
    done    = 0

    print(f"\n  Moving to home position ({TILT_HOME:.1f} deg) ...")
    pt.tilt_to(TILT_HOME, speed=SCAN_SPEED)
    time.sleep(ENDPOINT_PAUSE)

    print(f"\n  Moving to start position ({TILT_START:.1f} deg) ...")
    pt.tilt_to(TILT_START, speed=SCAN_SPEED)
    time.sleep(ENDPOINT_PAUSE)

    try:
        while endless or done < PASS_COUNT:
            done  += 1
            label  = "inf" if endless else f"{done}/{PASS_COUNT}"

            if pattern == "single":
                print(f"\n  Pass {label}: {TILT_START:.1f} -> {TILT_END:.1f} deg")
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
                print(f"\n  Pass {label}: {a:.1f} -> {b:.1f} deg")
                pt.tilt_to(b, speed=SCAN_SPEED)
                time.sleep(ENDPOINT_PAUSE)

            else:
                raise ValueError(f"Unknown SCAN_PATTERN: {pattern!r}")

    except KeyboardInterrupt:
        print("\n\n  Scan stopped by user (Ctrl+C).")
        pt.tilt_to(TILT_HOME, speed=SCAN_SPEED)

    else:
        print("\n  Scan complete -- returning to home position ...")
        pt.tilt_to(TILT_HOME, speed=SCAN_SPEED)

    pt.stop()
    final = pt.get_tilt()
    pan   = pt.get_pan()
    print(f"\n  Scan complete. Final -- Pan: {pan} deg  Tilt: {final} deg")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Pelco-D PT-3002 -- Sky Scan")
    print("=" * 60)
    print(f"  Port={PORT}  Baud={BAUDRATE}  Address={ADDRESS}")
    print(f"  Tilt range: {TILT_START:.1f} -> {TILT_END:.1f} deg")
    print(f"  Speed: {SCAN_SPEED}")
    print(f"  Pattern: {SCAN_PATTERN}")
    print(f"  Passes: {PASS_COUNT if PASS_COUNT > 0 else 'inf (Ctrl+C to stop)'}")
    print(f"  Inverted: {TILT_INVERTED}")
    print()

    try:
        pt = PT3002()
    except serial.SerialException as e:
        print(f"\n  ERROR: {e}")
        return

    with pt:
        run_sky_scan(pt)


if __name__ == "__main__":
    main()