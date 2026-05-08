"""
FastAPI WebSocket server for browser-based VisionBeam.

Browser captures frames from a user-selected camera, JPEG-encodes them,
and streams them over WebSocket. Each frame is decoded, run through the
tracker, and the detected boxes + chosen target pixel are returned as
JSON. When pixel-aim calibration is loaded and auto-aim is on, the server
also drives the DMX fixture.

Live control loop:
    HybridMethod.process_frame(frame) -> (u, v) in pixels
    PixelAimCalibration.predict(u, v) -> (pan_deg, tilt_deg)
    DMXConnection.aim(pan_deg, tilt_deg) -> DMX universe over USB

There is no floor-coordinate projection and no temporal smoothing between
the tracker's pixel output and DMX. The quadratic pixel-to-pan/tilt fit
is the only spatial transform on the live path.

Environment:
    VISIONBEAM_NO_DMX=1     run without DMX hardware (mock fixture)
    VISIONBEAM_DMX_PORT=... serial port (default /dev/ttyUSB0)
    VISIONBEAM_FIXTURE=...  path to fixture profile JSON

Control messages (text JSON, client -> server):
    {"type":"lock","track_id":7}
    {"type":"unlock"}
    {"type":"auto_aim","enabled":true|false}
    {"type":"aim","pan":270.0,"tilt":135.0}
    {"type":"calibrate_sample","pan":..,"tilt":..,"px":..|null,"py":..|null}
    {"type":"calibration_fit"}
    {"type":"calibration_clear"}

Detection results (server -> client, after each binary frame):
    {
      "tracks":      [{"id":7,"x1":..,"y1":..,"x2":..,"y2":..}, ...],
      "target_px":   [x, y] | null,
      "locked_id":   int | null,
      "frame_size":  [w, h],
      "auto_aim":    bool,
      "pan":         float | null,
      "tilt":        float | null,
      "calibration": {
        "n_samples":    int,
        "n_in_frame":   int,
        "fitted":       bool,
        "rms_pan_deg":  float | null,
        "rms_tilt_deg": float | null
      }
    }

Run:
    cd backend
    VISIONBEAM_NO_DMX=1 uvicorn server:app --reload
"""

from __future__ import annotations

import glob
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from visionbeam.aim import PixelAimCalibration
from visionbeam.dmx import DMXConnection, FixtureProfile, MockDMX
from visionbeam.tracker import HybridMethod

PAN_RANGE_DEFAULT = 540.0
TILT_RANGE_DEFAULT = 270.0


def _synthetic_beam(
    pan: float, tilt: float, w: int, h: int,
    pan_range: float, tilt_range: float,
) -> tuple[float, float]:
    """
    Hidden ground-truth pan/tilt -> pixel mapping for MockDMX.

    The point is to produce something with enough non-linearity that a
    naive linear "click and fit" would have measurable RMS error, so the
    quadratic calibration is actually doing work. Used purely for dev
    testing without a real fixture.
    """
    u = (pan / pan_range) * 2.0 - 1.0
    v = (tilt / tilt_range) * 2.0 - 1.0
    k = 0.15
    px = (u + k * u * v) * 0.45 * w + w / 2.0
    py = (v + k * v * v) * 0.4 * h + h / 2.0
    return px, py

logger = logging.getLogger("visionbeam.server")
logging.basicConfig(level=logging.INFO)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXTURE = os.path.join(HERE, "config", "fixture_zq02360_15ch.json")
CALIBRATION_PATH = os.path.join(HERE, "calibration", "aim.json")


def _autodetect_dmx_port() -> str | None:
    candidates = sorted(
        glob.glob("/dev/tty.usbserial-*")
        + glob.glob("/dev/tty.usbmodem*")
        + glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/ttyACM*")
    )
    return candidates[0] if candidates else None


class AppState:
    """Process-wide state shared across WebSocket connections."""
    dmx: DMXConnection | MockDMX | None = None
    calibration: PixelAimCalibration = PixelAimCalibration()


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    fixture_path = os.environ.get("VISIONBEAM_FIXTURE", DEFAULT_FIXTURE)
    fixture = FixtureProfile(fixture_path)

    if os.environ.get("VISIONBEAM_NO_DMX"):
        logger.info("DMX disabled (VISIONBEAM_NO_DMX set) — using MockDMX")
        state.dmx = MockDMX(fixture)
    else:
        port = os.environ.get("VISIONBEAM_DMX_PORT") or _autodetect_dmx_port()
        if port is None:
            logger.warning(
                "no USB-DMX adapter found (looked for /dev/tty.usbserial-*, "
                "/dev/tty.usbmodem*, /dev/ttyUSB*, /dev/ttyACM*); "
                "falling back to MockDMX"
            )
            state.dmx = MockDMX(fixture)
        else:
            try:
                logger.info("opening DMX on %s", port)
                state.dmx = DMXConnection(port, fixture)
                state.dmx.set_defaults(dimmer=255)
                state.dmx.start()
            except Exception as e:
                logger.warning(
                    "DMX open failed (%s); falling back to MockDMX. "
                    "Set VISIONBEAM_NO_DMX=1 to silence this warning.",
                    e,
                )
                state.dmx = MockDMX(fixture)

    state.calibration = PixelAimCalibration.load(CALIBRATION_PATH)
    logger.info("loaded calibration: %s", state.calibration.status())

    try:
        yield
    finally:
        if state.dmx is not None:
            try:
                state.dmx.blackout()
                state.dmx.stop()
            except Exception:
                logger.exception("error closing DMX")


app = FastAPI(title="VisionBeam", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "calibration": state.calibration.status()}


class Connection:
    """Per-WebSocket mutable state."""
    def __init__(self):
        self.tracker = HybridMethod()
        self.auto_aim = True


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    conn = Connection()
    logger.info("client connected")

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                payload = _handle_frame(msg["bytes"], conn)
                await websocket.send_json(payload)
                continue

            if "text" in msg and msg["text"] is not None:
                _handle_control(msg["text"], conn)
                continue

            if msg.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    finally:
        logger.info("client disconnected")


def _handle_frame(data: bytes, conn: Connection) -> dict[str, Any]:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "decode_failed"}

    target = conn.tracker.process_frame(frame)
    if target is not None:
        conn.tracker.set_beam_position(int(target[0]), int(target[1]))

    pan: float | None = None
    tilt: float | None = None
    if (
        conn.auto_aim
        and target is not None
        and state.calibration.fitted
        and state.dmx is not None
    ):
        predicted = state.calibration.predict(target[0], target[1])
        if predicted is not None:
            pan, tilt = predicted
            state.dmx.aim(pan, tilt)

    h, w = frame.shape[:2]

    is_mock = isinstance(state.dmx, MockDMX)
    beam_px: list[float] | None = None
    if is_mock and isinstance(state.dmx, MockDMX):
        beam_pan = pan if pan is not None else state.dmx.last_pan
        beam_tilt = tilt if tilt is not None else state.dmx.last_tilt
        if beam_pan is not None and beam_tilt is not None:
            fixture = state.dmx.fixture
            bx, by = _synthetic_beam(
                beam_pan, beam_tilt, w, h,
                fixture.pan_range if fixture else PAN_RANGE_DEFAULT,
                fixture.tilt_range if fixture else TILT_RANGE_DEFAULT,
            )
            beam_px = [bx, by]

    return {
        "tracks": [
            {"id": t.id, "x1": t.x1, "y1": t.y1, "x2": t.x2, "y2": t.y2}
            for t in conn.tracker.tracks
        ],
        "target_px": [target[0], target[1]] if target is not None else None,
        "locked_id": conn.tracker.locked_id,
        "frame_size": [w, h],
        "auto_aim": conn.auto_aim,
        "pan": pan,
        "tilt": tilt,
        "dmx_type": "mock" if is_mock else "real",
        "beam_px": beam_px,
        "calibration": state.calibration.status(),
    }


def _handle_control(text: str, conn: Connection) -> None:
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("bad control message: %r", text)
        return

    kind = msg.get("type")

    if kind == "lock":
        track_id = msg.get("track_id")
        if isinstance(track_id, int):
            conn.tracker.lock_to_id(track_id)
            logger.info("locked to id=%d", track_id)

    elif kind == "unlock":
        conn.tracker.lock_to_id(None)
        logger.info("unlocked")

    elif kind == "auto_aim":
        conn.auto_aim = bool(msg.get("enabled", True))
        logger.info("auto_aim=%s", conn.auto_aim)

    elif kind == "aim":
        pan = msg.get("pan")
        tilt = msg.get("tilt")
        if isinstance(pan, (int, float)) and isinstance(tilt, (int, float)):
            if state.dmx is not None:
                state.dmx.aim(float(pan), float(tilt))

    elif kind == "calibrate_sample":
        pan = msg.get("pan")
        tilt = msg.get("tilt")
        px = msg.get("px")
        py = msg.get("py")
        if isinstance(pan, (int, float)) and isinstance(tilt, (int, float)):
            state.calibration.add_sample(
                float(pan),
                float(tilt),
                float(px) if isinstance(px, (int, float)) else None,
                float(py) if isinstance(py, (int, float)) else None,
            )
            logger.info(
                "sample added (pan=%.1f tilt=%.1f px=%s py=%s) total=%d",
                pan, tilt, px, py, len(state.calibration.samples),
            )

    elif kind == "calibration_fit":
        result = state.calibration.fit()
        logger.info("fit result: %s", result)
        if state.calibration.fitted:
            state.calibration.save(CALIBRATION_PATH)

    elif kind == "calibration_clear":
        state.calibration.clear()
        if os.path.exists(CALIBRATION_PATH):
            os.remove(CALIBRATION_PATH)
        logger.info("calibration cleared")

    elif kind == "set_lamp":
        if state.dmx is None:
            return
        fixture_channels = (
            state.dmx.fixture.channels if state.dmx.fixture else {}
        )
        mapping = {
            "dimmer": "dimmer",
            "r": "red",
            "g": "green",
            "b": "blue",
            "w": "white",
        }
        for key, channel_name in mapping.items():
            value = msg.get(key)
            if isinstance(value, (int, float)) and channel_name in fixture_channels:
                state.dmx.set_channel(channel_name, int(value))
        logger.info(
            "set_lamp dimmer=%s r=%s g=%s b=%s w=%s",
            msg.get("dimmer"), msg.get("r"), msg.get("g"),
            msg.get("b"), msg.get("w"),
        )

    else:
        logger.warning("unknown control type: %r", kind)
