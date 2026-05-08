import { useEffect, useRef, useState } from "react";
import type { DetectionResult, Track } from "./types";

interface Props {
  stream: MediaStream | null;
  detection: DetectionResult | null;
  onClick: (px: number, py: number, hitId: number | null) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

export function Viewport({ stream, detection, onClick, videoRef }: Props) {
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);

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

  function toCanvasCoords(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = overlayRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const px = ((e.clientX - rect.left) * canvas.width) / rect.width;
    const py = ((e.clientY - rect.top) * canvas.height) / rect.height;
    return { px, py };
  }

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!detection) return;
    const c = toCanvasCoords(e);
    if (!c) return;
    const hit = hitTest(detection.tracks, c.px, c.py);
    onClick(c.px, c.py, hit);
  }

  function handleMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const c = toCanvasCoords(e);
    if (c) setHover({ x: c.px, y: c.py });
  }

  return (
    <div className="viewport">
      <video ref={videoRef} autoPlay playsInline muted />
      <canvas
        ref={overlayRef}
        onClick={handleClick}
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
      />
      <div className="viewport-frame">
        <span />
      </div>
      <div className="viewport-tag">live · feed</div>
      {hover && (
        <div className="viewport-coords">
          x {hover.x.toFixed(0)} · y {hover.y.toFixed(0)}
        </div>
      )}
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
    const color = isLocked ? "#ff3548" : "#00e5ff";
    const w = t.x2 - t.x1;
    const h = t.y2 - t.y1;

    // Glow
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = isLocked ? 22 : 12;
    ctx.lineWidth = isLocked ? 2.5 : 1.5;
    ctx.strokeStyle = color;

    // Bracket-corner box rather than full rectangle — lighting console feel
    const c = Math.min(20, w * 0.18, h * 0.18);
    ctx.beginPath();
    // top-left
    ctx.moveTo(t.x1, t.y1 + c); ctx.lineTo(t.x1, t.y1); ctx.lineTo(t.x1 + c, t.y1);
    // top-right
    ctx.moveTo(t.x2 - c, t.y1); ctx.lineTo(t.x2, t.y1); ctx.lineTo(t.x2, t.y1 + c);
    // bottom-right
    ctx.moveTo(t.x2, t.y2 - c); ctx.lineTo(t.x2, t.y2); ctx.lineTo(t.x2 - c, t.y2);
    // bottom-left
    ctx.moveTo(t.x1 + c, t.y2); ctx.lineTo(t.x1, t.y2); ctx.lineTo(t.x1, t.y2 - c);
    ctx.stroke();

    // Faint inner outline for locked
    if (isLocked) {
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 6]);
      ctx.strokeRect(t.x1, t.y1, w, h);
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    ctx.restore();

    // Label tag
    const label = isLocked ? `LOCKED · #${t.id}` : `TRK · #${t.id}`;
    ctx.font =
      '600 12px "JetBrains Mono", ui-monospace, Cascadia Code, monospace';
    const metrics = ctx.measureText(label);
    const padX = 8, padY = 5, labelH = 22;
    const tagW = metrics.width + padX * 2;
    const tagX = t.x1;
    const tagY = t.y1 - labelH - 2;

    ctx.fillStyle = "rgba(0,0,0,0.75)";
    ctx.fillRect(tagX, tagY, tagW, labelH);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(tagX + 0.5, tagY + 0.5, tagW - 1, labelH - 1);

    // Color stripe on left
    ctx.fillStyle = color;
    ctx.fillRect(tagX, tagY, 2, labelH);

    ctx.fillStyle = isLocked ? "#fff" : "#cdfbff";
    ctx.fillText(label, tagX + padX, tagY + labelH - padY - 1);
  }

  if (det.target_px) {
    const [tx, ty] = det.target_px;
    ctx.save();
    ctx.shadowColor = "#ffc83d";
    ctx.shadowBlur = 16;
    ctx.strokeStyle = "#ffc83d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(tx, ty, 16, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(tx - 22, ty); ctx.lineTo(tx - 8, ty);
    ctx.moveTo(tx + 8, ty);  ctx.lineTo(tx + 22, ty);
    ctx.moveTo(tx, ty - 22); ctx.lineTo(tx, ty - 8);
    ctx.moveTo(tx, ty + 8);  ctx.lineTo(tx, ty + 22);
    ctx.stroke();
    ctx.restore();
  }

  if (det.beam_px) {
    const [bx, by] = det.beam_px;
    // Soft outer halo
    const grad = ctx.createRadialGradient(bx, by, 4, bx, by, 60);
    grad.addColorStop(0, "rgba(0, 229, 255, 0.55)");
    grad.addColorStop(0.5, "rgba(0, 229, 255, 0.18)");
    grad.addColorStop(1, "rgba(0, 229, 255, 0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(bx, by, 60, 0, Math.PI * 2);
    ctx.fill();

    // Bright core
    ctx.save();
    ctx.shadowColor = "#00e5ff";
    ctx.shadowBlur = 24;
    ctx.fillStyle = "#e8fbff";
    ctx.beginPath();
    ctx.arc(bx, by, 6, 0, Math.PI * 2);
    ctx.fill();
    // Ring
    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(bx, by, 18, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = "#6ff3ff";
    ctx.font =
      '600 11px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillText("BEAM · SIM", bx + 24, by - 18);
  }
}
