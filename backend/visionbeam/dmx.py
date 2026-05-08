import json
import time
import threading
import serial

DMX_UNIVERSE_SIZE = 512
DMX_BAUD = 250000
DMX_BYTE_FORMAT = serial.EIGHTBITS
DMX_STOP_BITS = serial.STOPBITS_TWO
DMX_PARITY = serial.PARITY_NONE

BREAK_DURATION = 0.000092
MAB_DURATION = 0.000012


class FixtureProfile:
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
        return (self.start_address - 1) + (self.channels[name] - 1)


def angle_to_bytes(angle: float, total_range: float, invert: bool) -> tuple[int, int]:
    fraction = max(0.0, min(1.0, angle / total_range))
    if invert:
        fraction = 1.0 - fraction

    value = int(fraction * 65535)
    coarse = (value >> 8) & 0xFF
    fine = value & 0xFF
    return (coarse, fine)


class DMXConnection:
    def __init__(self, port: str, fixture: FixtureProfile):
        self.fixture = fixture
        self.frame = bytearray(DMX_UNIVERSE_SIZE + 1)
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

        self.port = serial.Serial(
            port=port,
            baudrate=DMX_BAUD,
            bytesize=DMX_BYTE_FORMAT,
            stopbits=DMX_STOP_BITS,
            parity=DMX_PARITY,
        )

    def set_channel(self, name: str, value: int):
        index = self.fixture.absolute_channel(name) + 1
        with self.lock:
            self.frame[index] = max(0, min(255, value))

    def set_defaults(self, **kwargs: int):
        for name, value in kwargs.items():
            self.set_channel(name, value)

    def aim(self, pan_deg: float, tilt_deg: float):
        pan_c, pan_f = angle_to_bytes(
            pan_deg, self.fixture.pan_range, self.fixture.pan_invert
        )
        tilt_c, tilt_f = angle_to_bytes(
            tilt_deg, self.fixture.tilt_range, self.fixture.tilt_invert
        )

        with self.lock:
            f = self.fixture
            self.frame[f.absolute_channel("pan_coarse") + 1] = pan_c
            self.frame[f.absolute_channel("pan_fine") + 1] = pan_f
            self.frame[f.absolute_channel("tilt_coarse") + 1] = tilt_c
            self.frame[f.absolute_channel("tilt_fine") + 1] = tilt_f

    def send_frame(self):
        self.port.break_condition = True
        time.sleep(BREAK_DURATION)
        self.port.break_condition = False
        time.sleep(MAB_DURATION)

        with self.lock:
            self.port.write(self.frame)

    def tx_loop(self):
        while self.running:
            self.send_frame()
            time.sleep(0.025)

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self.tx_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.port.close()

    def blackout(self):
        with self.lock:
            self.frame = bytearray(DMX_UNIVERSE_SIZE + 1)


class MockDMX:
    def __init__(self, fixture: FixtureProfile | None = None):
        self.fixture = fixture
        self.last_pan: float | None = None
        self.last_tilt: float | None = None

    def set_channel(self, name: str, value: int):
        pass

    def set_defaults(self, **kwargs: int):
        pass

    def aim(self, pan_deg: float, tilt_deg: float):
        self.last_pan = pan_deg
        self.last_tilt = tilt_deg

    def start(self):
        pass

    def stop(self):
        pass

    def blackout(self):
        self.last_pan = None
        self.last_tilt = None
