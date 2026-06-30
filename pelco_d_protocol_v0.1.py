# flake8: noqa
"""
Pelco-D PT-3002 — Sky Scan Script
==================================
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


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

PORT      = "COM4"
BAUDRATE  = 2400
ADDRESS   = 1
CMD_DELAY = 0.05

# Tilt angle limits (degrees) — adjust to your sky range
TILT_START = 40.0
TILT_END   = 90.0

# Movement speed: 0–63
#   6  = ~10% (~1°/sec tilt)
#  16  = ~25% (~2.5°/sec tilt)
#  32  = ~50% (~5°/sec tilt)
#  48  = ~75% (~7.5°/sec tilt)
#  63  = 100% (~10°/sec tilt, max)
SCAN_SPEED = 30

# "single"   — one pass from TILT_START to TILT_END, then stop
# "pendulum" — back-and-forth: START→END→START→END ...
SCAN_PATTERN = "pendulum"

# Number of passes (0 = infinite loop, stop with Ctrl+C)
PASS_COUNT = 4

# Seconds to pause at each endpoint before reversing
ENDPOINT_PAUSE = 0.5

# FIX 1: Tilt direction is inverted on this device.
# When the angle INCREASES we must send tilt_down, and vice versa.
# Set to True if tilt angle grows when moving "down", False otherwise.
TILT_INVERTED = True

# FIX 2: Overshoot tolerance — increased to avoid false triggers.
# The device may briefly report a position going the wrong way
# due to mechanical inertia. Only trigger if deviation > this value.
OVERSHOOT_THRESHOLD = 3.0   # degrees

# How close to target before stopping (degrees)
POSITION_TOLERANCE = 0.8

# Tilt angle for home/park position (returned to on start and exit)
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

    def __init__(self, port=PORT, address=ADDRESS,
                 baudrate=BAUDRATE, cmd_delay=CMD_DELAY):
        self.addr  = address
        self.delay = cmd_delay
        self.ser   = serial.Serial(
            port=port, baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5,
            dsrdtr=False,
            rtscts=False
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
        time.sleep(0.3)
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

    def get_tilt(self):
        """Returns current tilt in degrees, or None on timeout."""
        resp = self._query(_build_ext(self.addr, Opcode.QUERY_TILT_POS, 0, 0))
        if resp and len(resp) >= 7:
            raw = (resp[4] << 8) | resp[5]
            return (raw - 36000) / 100.0 if raw > 18000 else raw / 100.0
        return None

    def get_pan(self):
        """Returns current pan in degrees, or None on timeout."""
        resp = self._query(_build_ext(self.addr, Opcode.QUERY_PAN_POS, 0, 0))
        if resp and len(resp) >= 7:
            return ((resp[4] << 8) | resp[5]) / 100.0
        return None

    def tilt_to(self, target_deg, speed=SCAN_SPEED, timeout=40.0):
        """
        Moves tilt continuously toward target_deg and stops
        when position feedback is within POSITION_TOLERANCE degrees.
        """
        current = self.get_tilt()
        if current is None:
            print(f"    [WARNING] No position feedback.")
            return None

        if abs(current - target_deg) <= POSITION_TOLERANCE:
            return current

        going_up = target_deg > current
        if going_up:
            self._tilt_increasing(speed)
        else:
            self._tilt_decreasing(speed)

        # Poll faster at high speed to catch target before overshooting
        poll_interval = max(0.05, 0.15 - (speed / 63.0) * 0.10)

        t_start   = time.time()
        prev_pos  = current
        stable_wrong_dir = 0

        while True:
            time.sleep(poll_interval)
            pos     = self.get_tilt()
            elapsed = time.time() - t_start

            if pos is not None:
                remaining = abs(pos - target_deg)
                print(f"    Tilt: {pos:6.2f}°  →  target: {target_deg:.2f}°  "
                      f"(Δ {remaining:.2f}°)    ", end="\r")

                # Brake earlier at high speed to compensate for inertia
                brake_margin = POSITION_TOLERANCE + (speed / 63.0) * 3.0
                if remaining <= brake_margin:
                    self.stop()
                    print(f"    Tilt: {pos:6.2f}°  →  target: {target_deg:.2f}°")
                    return pos

                if prev_pos is not None:
                    moved_wrong = (going_up  and pos < prev_pos - OVERSHOOT_THRESHOLD) or \
                                  (not going_up and pos > prev_pos + OVERSHOOT_THRESHOLD)
                    if moved_wrong:
                        stable_wrong_dir += 1
                        if stable_wrong_dir >= 3:
                            self.stop()
                            print(f"\n    [WARNING] Consistent overshoot at {pos:.2f}°")
                            return pos
                    else:
                        stable_wrong_dir = 0

                prev_pos = pos

            if elapsed > timeout:
                self.stop()
                print(f"\n    [WARNING] Timeout after {timeout}s at {pos}°")
                return pos


# ── Sky Scan ──────────────────────────────────────────────────

def run_sky_scan(pt: PT3002):
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
            done += 1
            label = "∞" if endless else f"{done}/{PASS_COUNT}"

            if pattern == "single":
                print(f"\n  Pass {label}: {TILT_START:.1f}° → {TILT_END:.1f}°")
                pt.tilt_to(TILT_END, speed=SCAN_SPEED)
                time.sleep(ENDPOINT_PAUSE)
                if not endless and done >= PASS_COUNT:
                    break
                print(f"  Returning to start ...")
                pt.tilt_to(TILT_START, speed=SCAN_SPEED)
                time.sleep(ENDPOINT_PAUSE)

            elif pattern == "pendulum":
                a = TILT_START if done % 2 == 1 else TILT_END
                b = TILT_END   if done % 2 == 1 else TILT_START
                print(f"\n  Pass {label}: {a:.1f}° → {b:.1f}°")
                pt.tilt_to(b, speed=SCAN_SPEED)
                time.sleep(ENDPOINT_PAUSE)

            else:
                raise ValueError(f"Unknown SCAN_PATTERN: {pattern!r}")

    except KeyboardInterrupt:
        print("\n\n  Scan stopped by user (Ctrl+C).")
        pt.tilt_to(TILT_HOME, speed=SCAN_SPEED)
        
    else:
        print(f"\n  Scan complete — returning to home position ...")
        pt.tilt_to(TILT_HOME, speed=SCAN_SPEED)

    pt.stop()
    final = pt.get_tilt()
    pan   = pt.get_pan()
    print(f"\n  Scan complete. Final position — Pan: {pan}°  Tilt: {final}°")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Pelco-D PT-3002 — Sky Scan")
    print("=" * 60)
    print(f"  Port={PORT}  Baud={BAUDRATE}  Address={ADDRESS}")
    print(f"  Tilt range:  {TILT_START:.1f}° → {TILT_END:.1f}°")
    print(f"  Speed:       0x{SCAN_SPEED:02X}")
    print(f"  Pattern:     {SCAN_PATTERN}")
    print(f"  Passes:      {PASS_COUNT if PASS_COUNT > 0 else '∞ (Ctrl+C to stop)'}")
    print(f"  Inverted:    {TILT_INVERTED}")
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
