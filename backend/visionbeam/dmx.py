"""
DMX hardware interface.

Manages serial communication with a USB-to-DMX512 adapter
and maps logical channels to DMX addresses via fixture profiles.
"""

import json
import time
import threading
import serial

DMX_UNIVERSE_SIZE = 512
DMX_BAUD = 250000
DMX_BYTE_FORMAT = serial.EIGHTBITS
DMX_STOP_BITS = serial.STOPBITS_TWO
DMX_PARITY = serial.PARITY_NONE

BREAK_DURATION = 0.000092   # 92 µs minimum break
MAB_DURATION = 0.000012     # 12 µs mark-after-break


class FixtureProfile:
    """
    Loads a fixture's DMX channel map and physical parameters from JSON.
    """
    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = json.load(f)

        self.name: str = cfg["name"]
        self.start_address: int = cfg["dmx_start_address"]
        self.channels: dict[str, int] = cfg["channels"]
        self.pan_range: float = cfg["pan_range_deg"]
        self.tilt_range: float = cfg["tilt_range_deg"]
        self.pan_invert: bool = cfg["pan_invert"]
        self.tilt_invert: bool = cfg["tilt_invert"]

    def absolute_channel(self, name: str) -> int:
        """
        Returns the 0-indexed DMX channel for a logical channel name.
        """
        return (self.start_address - 1) + (self.channels[name] - 1)


def angle_to_bytes(angle: float, total_range: float, invert: bool) -> tuple[int, int]:
    """
    Converts a physical angle to 16-bit DMX coarse/fine bytes.

    Inputs:
        angle: Target angle in degrees, relative to the fixture's zero.
        total_range: The fixture's full range of motion in degrees.
        invert: If True, reverses the direction.

    Returns:
        (coarse, fine) byte tuple.
    """
    fraction = max(0.0, min(1.0, angle / total_range))
    if invert:
        fraction = 1.0 - fraction
    
    value = int(fraction * 65535)
    coarse = (value >> 8) & 0xFF
    fine = value & 0xFF
    return (coarse, fine)


class DMXConnection:
    """
    Manages a serial connection to a USB-to-DMX adapter (Enttec Open DMX Style).
    """
    def __init__(self, port: str, fixture: FixtureProfile):
        self.fixture = fixture
        self._frame = bytearray(DMX_UNIVERSE_SIZE + 1)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._port = serial.Serial(
            port=port,
            baudrate=DMX_BAUD,
            bytesize=DMX_BYTE_FORMAT,
            stopbits=DMX_STOP_BITS,
            parity=DMX_PARITY
        )

    def set_channel(self, name: str, value: int):
        """
        Sets a single logical channel by name (0-255).
        """
        index = self.fixture.absolute_channel(name) + 1
        with self._lock:
            self._frame[index] = max(0, min(255, value))

    def set_defaults(self, **kwargs: int):
        """
        Sets static channel values.
        ex. dimmer=255, color=0
        """
        for name, value in kwargs.items():
            self.set_channel(name, value)

    def aim(self, pan_deg: float, tilt_deg: float):
        """
        Set the fixture's pan/tilt from physical angles in degrees.
        """
        pan_c, pan_f = angle_to_bytes(
            pan_deg, self.fixture.pan_range, self.fixture.pan_invert
        )
        tilt_c, tilt_f = angle_to_bytes(
            tilt_deg, self.fixture.tilt_range, self.fixture.tilt_invert
        )

        with self._lock:
            f = self.fixture
            self._frame[f.absolute_channel("pan_coarse") + 1] = pan_c
            self._frame[f.absolute_channel("pan_fine") + 1] = pan_f
            self._frame[f.absolute_channel("tilt_coarse") + 1] = tilt_c
            self._frame[f.absolute_channel("tilt_fine") + 1] = tilt_f

    def _send_frame(self):
        """
        Transmits one DMX frame: break, mark-after-break, then 513 bytes.
        """
        self._port.break_condition = True
        time.sleep(BREAK_DURATION)
        self._port.break_condition = False
        time.sleep(MAB_DURATION)

        with self._lock:
            self._port.write(self._frame)

    def _tx_loop(self):
        """
        Continuously sends the current frame at ~40 Hz.
        """
        while self._running:
            self._send_frame()
            time.sleep(0.025)

    def start(self):
        """
        Begins transmitting DMX frames in a background thread.
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stops transmitting and closes the serial port.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._port.close()

    def blackout(self):
        """
        Zero out all channels (including dimmer).
        """
        with self._lock:
            self._frame = bytearray(DMX_UNIVERSE_SIZE + 1)
