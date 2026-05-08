# VisionBeam

![VisionBeam operator console in run mode, showing two tracked people and one locked target with simulated beam](console.png)

A vision-driven autonomous light framework that finds the most active person in a room and follows them in real time. It requires no body-worn beacons, pre-programmed cues, or specialized cameras. A webcam in the browser sends frames to a Python server running a hybrid YOLOv8 + ByteTrack + masked motion-heatmap tracker, which maps the target pixel to fixture pan/tilt over USB-to-DMX512.

Built for MIT 6.S058 (Introduction to Computer Vision) to evaluate whether combining a deep-learning person detector with classical motion analysis is more robust than either technique alone under hostile stage lighting.

## Project Structure

```
backend/
├── server.py                       FastAPI + WebSocket entry point
├── requirements.txt
├── visionbeam/
│   ├── tracker.py                  HybridMethod: YOLOv8 + ByteTrack + motion heatmap
│   ├── aim.py                      Quadratic pixel → pan/tilt calibration
│   └── dmx.py                      USB-to-DMX512 driver
├── evaluation/                     Offline eval: record, ground truth, metrics, figures
├── calibration/                    aim.json (gitignored, venue-specific)
└── config/                         DMX fixture profile

frontend/
├── src/
│   ├── App.tsx                     WebSocket lifecycle, run/calibrate modes
│   ├── Viewport.tsx                Live video + tracking overlay
│   └── CalibrationPanel.tsx        Pan/tilt sliders, sample/fit controls
└── package.json
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A webcam (720p is sufficient)
- *(Optional)* A DMX moving head + USB-to-DMX512 adapter

### 1. Backend

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

Start the server (no DMX hardware needed):

```bash
cd backend
VISIONBEAM_NO_DMX=1 uvicorn server:app --reload
```

#### Environment Variables

| Variable | Purpose |
|----------|---------|
| `VISIONBEAM_NO_DMX=1` | Use `MockDMX` — no hardware required, UI shows a synthetic beam dot |
| `VISIONBEAM_DMX_PORT=/dev/tty.usbserial-XXXX` | Override DMX serial port (otherwise auto-detected) |
| `VISIONBEAM_FIXTURE=path/to/profile.json` | Override fixture profile (default: `backend/config/fixture_zq02360_15ch.json`) |

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed Vite URL, grant camera access, and the live tracking overlay should appear. If the backend is on a different host:

```bash
VITE_WS_URL=ws://192.168.1.50:8000/ws/detect npm run dev
```

### 3. Calibration (live fixture only)

Calibration maps pixel coordinates to DMX pan/tilt via a quadratic fit. It is only needed when driving a real fixture.

1. Switch the UI to **Calibrate** mode.
2. Use the **Pan/Tilt** sliders to aim the fixture, then click where the beam lands in the camera view. Repeat for ≥ 6 points spread across the frame.
3. Click **Fit** — the server solves a least-squares fit and saves coefficients to `backend/calibration/aim.json` (gitignored, venue-specific).
4. Switch back to **Run** to start live tracking.

See `backend/evaluation/` for the offline evaluation harness used to reproduce the paper's results.
