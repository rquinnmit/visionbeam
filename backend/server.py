"""
FastAPI WebSocket server for browser-based VisionBeam.

Browser captures frames from a user-selected camera, JPEG-encodes them,
and streams them over WebSocket. Each frame is decoded, run through the
tracker, and the detected boxes + chosen target pixel are returned as JSON.

Control messages (text):
    {"type": "lock",   "track_id": 7}    -> lock onto person 7
    {"type": "unlock"}                    -> release lock

Detection results (sent after each binary frame):
    {
      "tracks":     [{"id":7,"x1":..,"y1":..,"x2":..,"y2":..}, ...],
      "target_px":  [x, y] | null,
      "locked_id":  int | null,
      "frame_size": [w, h]
    }

Run:
    cd backend
    uvicorn server:app --reload
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from visionbeam.tracker import HybridMethod

logger = logging.getLogger("visionbeam.server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="VisionBeam")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    tracker = HybridMethod()
    logger.info("client connected")

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                payload = await _handle_frame(msg["bytes"], tracker)
                await websocket.send_json(payload)
                continue

            if "text" in msg and msg["text"] is not None:
                _handle_control(msg["text"], tracker)
                continue

            if msg.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    finally:
        logger.info("client disconnected")


async def _handle_frame(data: bytes, tracker: HybridMethod) -> dict:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "decode_failed"}

    target = tracker.process_frame(frame)
    if target is not None:
        tracker.set_beam_position(int(target[0]), int(target[1]))

    h, w = frame.shape[:2]
    return {
        "tracks": [
            {"id": t.id, "x1": t.x1, "y1": t.y1, "x2": t.x2, "y2": t.y2}
            for t in tracker.tracks
        ],
        "target_px": [target[0], target[1]] if target is not None else None,
        "locked_id": tracker.locked_id,
        "frame_size": [w, h],
    }


def _handle_control(text: str, tracker: HybridMethod) -> None:
    import json
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("bad control message: %r", text)
        return

    kind = msg.get("type")
    if kind == "lock":
        track_id = msg.get("track_id")
        if isinstance(track_id, int):
            tracker.lock_to_id(track_id)
            logger.info("locked to id=%d", track_id)
    elif kind == "unlock":
        tracker.lock_to_id(None)
        logger.info("unlocked")
    else:
        logger.warning("unknown control type: %r", kind)
