import { useCallback, useEffect, useRef, useState } from "react";
import { captureJpeg, listVideoDevices, openCamera, stopStream } from "./camera";
import { Viewport } from "./Viewport";
import type { ControlMessage, DetectionResult } from "./types";

const WS_URL =
  (import.meta.env.VITE_WS_URL as string | undefined) ??
  "ws://127.0.0.1:8000/ws/detect";

type WsState = "connecting" | "open" | "closed";

export default function App() {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [wsState, setWsState] = useState<WsState>("connecting");
  const [detection, setDetection] = useState<DetectionResult | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(
    typeof document !== "undefined" ? document.createElement("canvas") : null!,
  );
  const inFlightRef = useRef(false);

  // Initial: request a generic stream (unlocks device labels), then enumerate.
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
        const activeTrack = initial.getVideoTracks()[0];
        const activeId = activeTrack?.getSettings().deviceId;
        if (activeId) setDeviceId(activeId);
      } catch (err) {
        console.error("camera init failed", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Switch streams when the user picks a different camera.
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

  // Open the WebSocket once.
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

  // Kick the loop when both stream + ws are ready.
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

  const handleLock = useCallback(
    (trackId: number | null) => {
      if (trackId === null) {
        sendControl({ type: "unlock" });
        setDetection((d) => (d ? { ...d, locked_id: null } : d));
      } else {
        sendControl({ type: "lock", track_id: trackId });
        setDetection((d) => (d ? { ...d, locked_id: trackId } : d));
      }
    },
    [sendControl],
  );

  return (
    <div className="app">
      <div className="controls">
        <label>
          Camera:{" "}
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
          ws: {wsState}
        </span>
        {detection?.locked_id != null && (
          <button onClick={() => handleLock(null)}>Unlock #{detection.locked_id}</button>
        )}
      </div>
      <Viewport
        stream={stream}
        detection={detection}
        onLock={handleLock}
        videoRef={videoRef}
      />
    </div>
  );
}
