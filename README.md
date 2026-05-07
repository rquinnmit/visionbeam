# VisionBeam

## Introduction

VisionBeam is a spatially-aware autonomous party light that combines computer vision, spatial mapping, and live-event hardware. Rather than relying on pre-programmed lighting cues, the system uses person detection and multi-person tracking to identify who is dancing most and steers a moving head light to follow the action in real time.

This project also serves as a research study for MIT 6.S058, investigating how different computer vision approaches to motion-driven target selection perform under dynamic stage illumination. The core research question: **does combining deep learning detection with classical motion analysis (VisionBeam's hybrid approach) outperform either technique alone when lighting conditions are hostile?**

## Project Structure

```
visionbeam/               # Live system
├── calibration.py         # ArUco homography + light triangulation
├── ik.py                  # Floor-to-pan/tilt spatial translation + EMA smoothing
├── dmx.py                 # USB-to-DMX512 serial interface + fixture profiles
├── tracker.py             # Core hybrid method (YOLOv8 + ByteTrack + masked motion heatmap)
├── pipeline.py            # Threaded camera → tracker → IK → DMX loop
└── ui.py                  # PySide6 Director's Station UI

evaluation/                # Research evaluation framework
├── methods.py             # Baseline target-selection methods (frame diff, Farneback, detection-only)
├── record.py              # Dataset recording tool (4 lighting conditions)
├── ground_truth.py        # Tracking-marker ground truth extraction
├── evaluate.py            # Metrics harness (accuracy, jitter, robustness, throughput)
└── visualize.py           # Matplotlib figure generation for the report

config/
└── fixture_default.json   # Generic moving head DMX channel map
```

## Getting Started

### 1. Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run that loads YOLO may download `yolov8n.pt`.

### 2. Calibration files

The code loads **`calibration/homography.json`** and **`calibration/mount.json`** (see defaults in `main.py` and `evaluation.evaluate`). Those paths are **gitignored** so each venue’s measured values stay on your machine.

**Committed templates** (safe to share, not tied to a real room):

| Template | Role |
|----------|------|
| [`calibration/homography.example.json`](calibration/homography.example.json) | **Shape and format** that [`FloorCalibration.save`](visionbeam/calibration.py) / `load` expect: a single key `"homography"` whose value is a **3×3** matrix. The committed matrix is the **identity**. That makes JSON valid and lets you run the app once you copy the file, but **identity does not model a real camera** — pixel coordinates would map naïvely to floor coordinates and metrics would be meaningless until you replace this with a homography computed from four or more measured pixel ↔ floor point pairs. |
| [`calibration/mount.example.json`](calibration/mount.example.json) | **Shape and format** that [`LightMount.save`](visionbeam/ik.py) / `load` expect: fixture position **x, y, z** (meters on the floor plane, **z** = height), plus **pan_offset** and **tilt_offset** (degrees) linking world angles to DMX. The numbers are **placeholders** (e.g. fixture near origin, **z = 3 m**, defaults matching `LightMount`). Replace after **`triangulate_light(...)`** from measured aim points or manual survey. |

**First-time setup:** copy templates to the filenames the programs read, then overwrite with real calibration when you have measurements.

```bash
cp calibration/homography.example.json calibration/homography.json
cp calibration/mount.example.json calibration/mount.json
```

See **System Pipeline → Spatial Calibration** below for how to measure ArUco floor points and light aim points and generate the real JSON via `FloorCalibration` / `triangulate_light`.

### 3. Smoke test: live pipeline (no DMX)

Runs camera \(\rightarrow\) hybrid tracker \(\rightarrow\) IK. No USB-DMX adapter required.

```bash
python main.py --no-dmx --camera 0
```

Press **`q`** to quit, **`a`** to toggle AUTO vs MANUAL (manual click-to-aim is planned for the PySide UI; OpenCV preview only shows tracking).

Use **`--camera 1`** (etc.) if the default device is wrong. On macOS, DMX serial ports are often `/dev/tty.usbserial-*` rather than Linux’s `/dev/ttyUSB0`; pass **`--dmx-port`** when using hardware.

### 4. Research evaluation (offline)

Record clips, extract marker-based ground truth, run metrics, then plot figures.

```bash
mkdir -p data/clips data/gt results figures

# Record one clip per lighting condition (interactive prompts between takes)
python -m evaluation.record --camera 0 --output data/clips --duration 30

# Ground truth: match video basename → data/gt/<basename>_gt.csv
python -m evaluation.ground_truth \
  --video data/clips/<your_clip>.mp4 \
  --mode color \
  --output data/gt \
  --preview
```

Tune **`--hsv-low`** and **`--hsv-high`** (comma-separated `H,S,V`) so the colored marker tracks reliably, then re-run without **`--preview`** for the final CSV.

```bash
python -m evaluation.evaluate \
  --clips data/clips \
  --gt data/gt \
  --calibration calibration/homography.json \
  --output results

python -m evaluation.visualize --results results --output figures
```

Optional: **`--trajectory-clip <stem>`** on `visualize` (filename without `.mp4`) for a specific trajectory plot.

**Naming:** For each `data/clips/foo.mp4`, ground truth must be `data/gt/foo_gt.csv`. `evaluate` skips clips with no matching GT file.

### 5. Documentation links

* [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) — lighting conditions, metrics, figures  
* [`REFERENCES.md`](REFERENCES.md) — papers and benchmarks  

## System Pipeline

### 1. Spatial Calibration (one-time, pre-event)
Two-step calibration performed once at the venue before an event:
* **Camera-to-floor homography:** Four ArUco markers are placed at known positions on the floor. OpenCV detects them automatically and computes a homography matrix (`cv2.findHomography`) mapping camera pixels to top-down floor coordinates.
* **Light position triangulation:** The light is aimed at 3+ known floor points and the pan/tilt angles are recorded at each. A nonlinear least-squares solver (`scipy.optimize.least_squares`) triangulates the light's 3D mount position (x, y, z) in the same floor coordinate system.

### 2. Person Detection
YOLOv8-nano produces bounding boxes for every person in the scene. Non-human motion (doors, curtains, fog machines, reflections) is discarded at this stage, so downstream tracking operates exclusively on people. Detection can run every Nth frame (e.g., every 2nd or 3rd) to maintain throughput, with the tracker's Kalman predictions filling the gaps.

### 3. Multi-Person Tracking
ByteTrack associates detections across frames using a Kalman filter and IoU-based matching, assigning each person a persistent ID. This enables identity-aware behaviors: the system can follow a specific dancer, smoothly hand off between individuals, and ignore people who have been stationary.

### 4. Motion Heatmap & Target Selection
Frame differencing (`cv2.absdiff`) on downscaled grayscale frames (~320x240) produces a per-pixel motion signal. This signal is masked to only include regions covered by tracked person bounding boxes, then Gaussian-blurred to form a spatial heatmap. The peak of the heatmap identifies the most active dancer on the floor.

**Beam masking:** A mask is applied around the light's current aim point before computing motion, preventing the system from chasing its own beam.

**Temporal smoothing:** An exponential moving average on the target coordinates prevents frame-to-frame jitter and produces smooth light sweeps.

### 5. Spatial Translation (Floor → Pan/Tilt)
Given the light's known mount position from calibration, converting a floor target to pan/tilt angles is direct trigonometry (`atan2`). The homography first maps the heatmap peak from camera pixels to floor coordinates, then the IK step computes the required pan and tilt to aim from the light's mount point to that floor position.

### 6. DMX Hardware Actuation
Pan/tilt angles are scaled to DMX channel values (0–255 coarse, 0–255 fine for 16-bit resolution) and transmitted over USB-to-DMX512 serial at ~40 FPS. A fixture profile config maps logical channels (pan, tilt, dimmer, color, etc.) to DMX addresses, supporting different moving head models.

### 7. Director's Station UI
A real-time operator interface built initially with OpenCV's `imshow` for prototyping, with a planned migration to PySide6 for production use.

**Live monitoring view:**
* Warped top-down floor plan with motion heatmap overlay
* Current light aim and smoothed target indicators
* Camera and DMX connection status

**Manual override:** Click-to-aim on the floor plan, with configurable behavior (permanent override until auto is re-enabled, or timed return to autonomous mode).

**Calibration wizard:** Guided step-by-step UI for the ArUco homography and light triangulation setup.

**Architecture:** The pipeline (camera → CV → IK → DMX) runs on a dedicated thread, pushing display frames and metadata via `queue.Queue` to the UI thread. Manual overrides and parameter changes flow back through shared state.

## Research Evaluation

### Methods Compared

| Method | Detection | Motion Signal | Location |
|---|---|---|---|
| Frame Differencing | None | Classical (full-frame `absdiff`) | `evaluation/methods.py` |
| Farneback Dense Flow | None | Classical (dense optical flow) | `evaluation/methods.py` |
| Detection Only | DL (YOLOv8n + ByteTrack) | None | `evaluation/methods.py` |
| **Hybrid (VisionBeam)** | **DL (YOLOv8n + ByteTrack)** | **Classical (bbox-masked `absdiff`)** | `visionbeam/tracker.py` |

### Evaluation Protocol

Video clips are recorded in a controlled studio under 4 lighting conditions: ambient, external light (static), external light (dynamic), and fixture + external (dynamic). A bright tracking marker worn by the subject provides per-frame ground truth via offline color thresholding. Each method is run on every clip; predictions are transformed to floor coordinates via the calibrated homography and compared to ground truth.

### Metrics

* **Targeting accuracy** — mean Euclidean error on the floor plane (meters)
* **Target stability (jitter)** — total predicted-target path length per second
* **Robustness** — accuracy degradation from ambient to fixture + external (dynamic) conditions
* **Throughput** — FPS on evaluation hardware

See [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) for the full evaluation framework and [`REFERENCES.md`](REFERENCES.md) for related work.

## Hardware Requirements
1. **DMX Lighting Fixture:** 1x moving head light with DMX512 pan/tilt control (16-bit recommended)
2. **Communication Interface:** 1x USB-to-DMX512 adapter (e.g., Enttec Open DMX)
3. **Camera:** 1x standard USB webcam (720p sufficient)
4. **Calibration Markers:** 4x printed ArUco markers (generated via OpenCV)
5. **Computer:** Any modern laptop (a discrete or integrated GPU is recommended for real-time YOLO inference but not strictly required — CPU inference at ~25-30 FPS is sufficient)

## Software Dependencies
* `python 3.10+`
* `opencv-contrib-python` — camera capture, ArUco detection, homography, optical flow, frame differencing
* `ultralytics` — YOLOv8-nano person detection and ByteTrack multi-object tracking
* `torch` — PyTorch runtime required by ultralytics (CPU or CUDA)
* `scipy` — light position triangulation via least-squares optimization
* `pyserial` — USB-to-DMX512 serial communication
* `PySide6` — Director's Station UI
* `matplotlib` — evaluation figure generation
