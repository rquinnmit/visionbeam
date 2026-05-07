"""
Interactive DMX channel probe.

Auto-detects the USB-DMX adapter, holds pan/tilt at center, then lets you
type channel/value pairs to figure out which channel actually drives the
lamp, color, and shutter on your fixture.

Usage:
    cd backend
    python dmx_probe.py

Commands:
    1 255           set channel 1 to 255
    1 0             set channel 1 to 0
    all 255         set every channel 1..15 to 255
    all 0           blackout
    show            print current frame values for channels 1..15
    pan 270         convenience: aim pan at 270 degrees
    tilt 135        convenience: aim tilt at 135 degrees
    quit            exit
"""

import glob
import sys
import time

from visionbeam.dmx import DMXConnection, FixtureProfile, angle_to_bytes


def autodetect_port() -> str:
    candidates = sorted(
        glob.glob("/dev/tty.usbserial-*")
        + glob.glob("/dev/tty.usbmodem*")
        + glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/ttyACM*")
    )
    if not candidates:
        print("No USB-DMX adapter found.", file=sys.stderr)
        sys.exit(1)
    return candidates[0]


def main():
    fixture = FixtureProfile("config/fixture_zq02360_15ch.json")
    port = autodetect_port()
    print(f"Opening DMX on {port}")
    dmx = DMXConnection(port, fixture)
    dmx.start()

    # Center pan/tilt so the head doesn't slam against a stop while you probe.
    pan_c, pan_f = angle_to_bytes(270.0, fixture.pan_range, fixture.pan_invert)
    tilt_c, tilt_f = angle_to_bytes(135.0, fixture.tilt_range, fixture.tilt_invert)
    dmx._frame[1] = pan_c
    dmx._frame[2] = pan_f
    dmx._frame[3] = tilt_c
    dmx._frame[4] = tilt_f

    print("Holding pan=270, tilt=135. Probe channels 5..15 to find lamp/color/shutter.")
    print("Type 'quit' to exit.\n")

    try:
        while True:
            line = input("> ").strip().lower()
            if not line:
                continue
            if line in ("quit", "exit", "q"):
                break

            parts = line.split()

            if parts[0] == "show":
                for ch in range(1, 16):
                    print(f"  CH{ch:>2}: {dmx._frame[ch]}")
                continue

            if parts[0] == "all" and len(parts) == 2:
                value = max(0, min(255, int(parts[1])))
                for ch in range(1, 16):
                    dmx._frame[ch] = value
                print(f"  set CH1..15 = {value}")
                continue

            if parts[0] == "pan" and len(parts) == 2:
                deg = float(parts[1])
                c, f = angle_to_bytes(deg, fixture.pan_range, fixture.pan_invert)
                dmx._frame[1], dmx._frame[2] = c, f
                print(f"  pan = {deg} (CH1={c}, CH2={f})")
                continue

            if parts[0] == "tilt" and len(parts) == 2:
                deg = float(parts[1])
                c, f = angle_to_bytes(deg, fixture.tilt_range, fixture.tilt_invert)
                dmx._frame[3], dmx._frame[4] = c, f
                print(f"  tilt = {deg} (CH3={c}, CH4={f})")
                continue

            if len(parts) == 2 and parts[0].isdigit():
                ch = int(parts[0])
                value = max(0, min(255, int(parts[1])))
                if 1 <= ch <= 512:
                    dmx._frame[ch] = value
                    print(f"  CH{ch} = {value}")
                else:
                    print("  channel must be 1..512")
                continue

            print("  ?  expected: <ch> <value> | all <value> | pan <deg> | tilt <deg> | show | quit")
    finally:
        print("\nblackout + closing")
        dmx.blackout()
        time.sleep(0.1)
        dmx.stop()


if __name__ == "__main__":
    main()
