import { useEffect, useRef } from "react";
import type { DetectionResult, Track } from "./types";

interface Props {
  stream: MediaStream | null;
  detection: DetectionResult | null;
  onLock: (trackId: number | null) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

export function Viewport({ stream, detection, onLock, videoRef }: Props) {
  const overlayRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.srcObject !== stream) video.srcObject = stream;
  }, [stream, videoRef]);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas || !detection) return;
    const [w, h] = detection.frame_size;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    drawOverlay(canvas, detection);
  }, [detection]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = overlayRef.current;
    if (!canvas || !detection) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * canvas.width) / rect.width;
    const y = ((e.clientY - rect.top) * canvas.height) / rect.height;
    const hit = hitTest(detection.tracks, x, y);
    if (hit !== null) {
      onLock(detection.locked_id === hit ? null : hit);
    } else {
      onLock(null);
    }
  }

  return (
    <div className="viewport">
      <video ref={videoRef} autoPlay playsInline muted />
      <canvas ref={overlayRef} onClick={handleClick} />
    </div>
  );
}

function hitTest(tracks: Track[], x: number, y: number): number | null {
  for (const t of tracks) {
    if (x >= t.x1 && x <= t.x2 && y >= t.y1 && y <= t.y2) return t.id;
  }
  return null;
}

function drawOverlay(canvas: HTMLCanvasElement, det: DetectionResult) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const t of det.tracks) {
    const isLocked = t.id === det.locked_id;
    ctx.lineWidth = isLocked ? 4 : 2;
    ctx.strokeStyle = isLocked ? "#ff3b30" : "#00ff88";
    ctx.strokeRect(t.x1, t.y1, t.x2 - t.x1, t.y2 - t.y1);

    const label = `#${t.id}${isLocked ? " LOCKED" : ""}`;
    ctx.font = "16px ui-sans-serif, system-ui, sans-serif";
    const metrics = ctx.measureText(label);
    const padding = 4;
    const labelH = 20;
    ctx.fillStyle = isLocked ? "#ff3b30" : "#00ff88";
    ctx.fillRect(t.x1, t.y1 - labelH, metrics.width + padding * 2, labelH);
    ctx.fillStyle = "#000";
    ctx.fillText(label, t.x1 + padding, t.y1 - 5);
  }

  if (det.target_px) {
    const [tx, ty] = det.target_px;
    ctx.strokeStyle = "#ffcc00";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(tx, ty, 14, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(tx - 18, ty);
    ctx.lineTo(tx + 18, ty);
    ctx.moveTo(tx, ty - 18);
    ctx.lineTo(tx, ty + 18);
    ctx.stroke();
  }
}
