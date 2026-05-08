import { useCallback, useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { CalibrationPanel } from "./CalibrationPanel";
import { captureJpeg, listVideoDevices, openCamera, stopStream } from "./camera";
import { Viewport } from "./Viewport";
import type { ControlMessage, DetectionResult, Mode } from "./types";

const WS_URL =
  (import.meta.env.VITE_WS_URL as string | undefined) ??
  "ws://127.0.0.1:8000/ws/detect";

const PAN_RANGE = 540;
const TILT_RANGE = 270;
const DEFAULT_PAN = PAN_RANGE / 2;
const DEFAULT_TILT = TILT_RANGE / 2;

type WsState = "connecting" | "open" | "closed";

export default function App() {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [wsState, setWsState] = useState<WsState>("connecting");
  const [detection, setDetection] = useState<DetectionResult | null>(null);
  const [mode, setMode] = useState<Mode>("run");
  const [calPan, setCalPan] = useState(DEFAULT_PAN);
  const [calTilt, setCalTilt] = useState(DEFAULT_TILT);

  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(
    typeof document !== "undefined" ? document.createElement("canvas") : null!,
  );
  const inFlightRef = useRef(false);

  const appRef = useRef<HTMLDivElement | null>(null);
  const headerRef = useRef<HTMLDivElement | null>(null);
  const controlsRef = useRef<HTMLDivElement | null>(null);
  const viewportWrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!appRef.current) return;
    const ctx = gsap.context(() => {
      gsap.set([headerRef.current, controlsRef.current, viewportWrapRef.current], {
        opacity: 0,
        y: 14,
      });
      const tl = gsap.timeline({ defaults: { ease: "expo.out" } });
      tl.to(headerRef.current, { opacity: 1, y: 0, duration: 0.9 })
        .to(controlsRef.current, { opacity: 1, y: 0, duration: 0.7 }, "-=0.55")
        .to(viewportWrapRef.current, { opacity: 1, y: 0, duration: 0.9 }, "-=0.45");
    }, appRef);
    return () => ctx.revert();
  }, []);

  const prevLockRef = useRef<number | null>(null);
  useEffect(() => {
    const id = detection?.locked_id ?? null;
    if (id !== null && id !== prevLockRef.current && viewportWrapRef.current) {
      gsap.fromTo(
        viewportWrapRef.current,
        { filter: "brightness(1.6) saturate(1.4)" },
        { filter: "brightness(1) saturate(1)", duration: 0.55, ease: "power3.out" },
      );
    }
    prevLockRef.current = id;
  }, [detection?.locked_id]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const initial = await openCamera(null);
        if (cancelled) {
          stopStream(initial);
          return;
        }
        setStream(initial);
        const list = await listVideoDevices();
        if (cancelled) return;
        setDevices(list);
        const activeId = initial.getVideoTracks()[0]?.getSettings().deviceId;
        if (activeId) setDeviceId(activeId);
      } catch (err) {
        console.error("camera init failed", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!deviceId) return;
    const current = stream?.getVideoTracks()[0]?.getSettings().deviceId;
    if (current === deviceId) return;
    let cancelled = false;
    (async () => {
      try {
        const next = await openCamera(deviceId);
        if (cancelled) {
          stopStream(next);
          return;
        }
        stopStream(stream);
        setStream(next);
      } catch (err) {
        console.error("camera switch failed", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [deviceId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    setWsState("connecting");

    ws.addEventListener("open", () => setWsState("open"));
    ws.addEventListener("close", () => setWsState("closed"));
    ws.addEventListener("error", () => setWsState("closed"));
    ws.addEventListener("message", (e) => {
      inFlightRef.current = false;
      try {
        const data = JSON.parse(e.data) as DetectionResult;
        setDetection(data);
      } catch {
        /* ignore */
      }
      void sendNextFrame();
    });

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const sendNextFrame = useCallback(async () => {
    const ws = wsRef.current;
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video || !canvas) return;
    if (inFlightRef.current) return;
    const blob = await captureJpeg(video, canvas);
    if (!blob) return;
    inFlightRef.current = true;
    ws.send(await blob.arrayBuffer());
  }, []);

  useEffect(() => {
    if (wsState !== "open" || !stream) return;
    const video = videoRef.current;
    if (!video) return;
    const onCanPlay = () => void sendNextFrame();
    video.addEventListener("loadeddata", onCanPlay);
    if (video.readyState >= 2) void sendNextFrame();
    return () => video.removeEventListener("loadeddata", onCanPlay);
  }, [wsState, stream, sendNextFrame]);

  const sendControl = useCallback((msg: ControlMessage) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(msg));
  }, []);

  useEffect(() => {
    if (wsState !== "open") return;
    sendControl({ type: "set_lamp", dimmer: 255, r: 0, g: 0, b: 0, w: 255 });
  }, [wsState, sendControl]);

  const setLockOptimistic = useCallback((id: number | null) => {
    setDetection((d) => (d ? { ...d, locked_id: id } : d));
  }, []);

  const handleViewportClick = useCallback(
    (px: number, py: number, hitId: number | null) => {
      if (mode === "calibrate") {
        sendControl({
          type: "calibrate_sample",
          pan: calPan,
          tilt: calTilt,
          px,
          py,
        });
        return;
      }
      if (hitId === null) {
        sendControl({ type: "unlock" });
        setLockOptimistic(null);
      } else if (detection?.locked_id === hitId) {
        sendControl({ type: "unlock" });
        setLockOptimistic(null);
      } else {
        sendControl({ type: "lock", track_id: hitId });
        setLockOptimistic(hitId);
      }
    },
    [mode, calPan, calTilt, detection?.locked_id, sendControl, setLockOptimistic],
  );

  const handleModeChange = useCallback(
    (next: Mode) => {
      setMode(next);
      sendControl({ type: "auto_aim", enabled: next === "run" });
      sendControl({ type: "set_lamp", dimmer: 255, r: 0, g: 0, b: 0, w: 255 });
      if (next === "calibrate") {
        sendControl({ type: "aim", pan: calPan, tilt: calTilt });
      }
    },
    [calPan, calTilt, sendControl],
  );

  const handlePanChange = useCallback(
    (pan: number) => {
      setCalPan(pan);
      sendControl({ type: "aim", pan, tilt: calTilt });
    },
    [calTilt, sendControl],
  );

  const handleTiltChange = useCallback(
    (tilt: number) => {
      setCalTilt(tilt);
      sendControl({ type: "aim", pan: calPan, tilt });
    },
    [calPan, sendControl],
  );

  const handleOffScreen = useCallback(() => {
    sendControl({
      type: "calibrate_sample",
      pan: calPan,
      tilt: calTilt,
      px: null,
      py: null,
    });
  }, [calPan, calTilt, sendControl]);

  const handleFit = useCallback(() => {
    sendControl({ type: "calibration_fit" });
  }, [sendControl]);

  const handleClear = useCallback(() => {
    sendControl({ type: "calibration_clear" });
  }, [sendControl]);

  const locked = detection?.locked_id != null;

  return (
    <div
      className="app"
      ref={appRef}
      data-mode={mode}
      data-locked={locked ? "true" : "false"}
    >
      <header className="console-header" ref={headerRef}>
        <div className="brand">
          <span className="brand-mark-wrap">
            <svg className="brand-mark" viewBox="0 0 64 64" aria-hidden="true">
              <defs>
                <linearGradient id="vb-ring" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#ff2e93" />
                  <stop offset="100%" stopColor="#00e5ff" />
                </linearGradient>
              </defs>
              <circle cx="32" cy="32" r="20" fill="none" stroke="url(#vb-ring)" strokeWidth="2" />
              <circle cx="32" cy="32" r="13" fill="none" stroke="#00e5ff" strokeWidth="1" opacity="0.65" />
              <circle cx="32" cy="32" r="3.2" fill="#ff2e93" />
              <g stroke="#f5f1ff" strokeWidth="1" strokeLinecap="square" opacity="0.85">
                <line x1="32" y1="6" x2="32" y2="12" />
                <line x1="32" y1="52" x2="32" y2="58" />
                <line x1="6" y1="32" x2="12" y2="32" />
                <line x1="52" y1="32" x2="58" y2="32" />
              </g>
            </svg>
          </span>
          <div className="brand-text">
            <span className="brand-title">VISIONBEAM</span>
            <span className="brand-sub">stage tracking · console v0.1</span>
          </div>
        </div>
        <div />
        <div className="console-meta">
          <div className="meta-block">
            <span>mode</span>
            <span className="meta-val">{mode === "run" ? "PERFORM" : "CALIBRATE"}</span>
          </div>
          <div className="meta-block">
            <span>fixture</span>
            <span className="meta-val">
              {detection?.dmx_type ? detection.dmx_type.toUpperCase() : "—"}
            </span>
          </div>
          <div className="meta-block">
            <span>aim</span>
            <span className="meta-val">
              {detection?.pan != null && detection?.tilt != null
                ? `${detection.pan.toFixed(1)}° / ${detection.tilt.toFixed(1)}°`
                : "—"}
            </span>
          </div>
        </div>
      </header>

      <div className="controls" ref={controlsRef}>
        <label className="field">
          <span>cam</span>
          <select
            value={deviceId ?? ""}
            onChange={(e) => setDeviceId(e.target.value || null)}
          >
            {devices.length === 0 && <option value="">(no cameras)</option>}
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Camera ${d.deviceId.slice(0, 6)}`}
              </option>
            ))}
          </select>
        </label>

        <span className={`status ${wsState === "open" ? "connected" : "disconnected"}`}>
          ws · {wsState}
        </span>

        <div className="mode-toggle">
          <button
            className={mode === "run" ? "active" : ""}
            onClick={() => handleModeChange("run")}
          >
            Run
          </button>
          <button
            className={mode === "calibrate" ? "active" : ""}
            onClick={() => handleModeChange("calibrate")}
          >
            Calibrate
          </button>
        </div>

        {locked && (
          <span className="readout">
            target lock <strong>#{detection?.locked_id}</strong>
          </span>
        )}

        {mode === "run" && locked && (
          <button
            className="btn-unlock"
            onClick={() => {
              sendControl({ type: "unlock" });
              setLockOptimistic(null);
            }}
          >
            ✕ release #{detection?.locked_id}
          </button>
        )}
      </div>

      {mode === "calibrate" && detection && (
        <CalibrationPanel
          pan={calPan}
          tilt={calTilt}
          panRange={PAN_RANGE}
          tiltRange={TILT_RANGE}
          status={detection.calibration}
          onPanChange={handlePanChange}
          onTiltChange={handleTiltChange}
          onOffScreen={handleOffScreen}
          onFit={handleFit}
          onClear={handleClear}
        />
      )}

      <div className="viewport-wrap" ref={viewportWrapRef}>
        <Viewport
          stream={stream}
          detection={detection}
          onClick={handleViewportClick}
          videoRef={videoRef}
        />
      </div>
    </div>
  );
}
