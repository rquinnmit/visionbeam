export interface Track {
  id: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectionResult {
  tracks: Track[];
  target_px: [number, number] | null;
  locked_id: number | null;
  frame_size: [number, number];
}

export type ControlMessage =
  | { type: "lock"; track_id: number }
  | { type: "unlock" };
