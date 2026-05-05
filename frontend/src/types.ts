export interface Track {
  id: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface CalibrationStatus {
  n_samples: number;
  n_in_frame: number;
  fitted: boolean;
  rms_pan_deg: number | null;
  rms_tilt_deg: number | null;
}

export interface DetectionResult {
  tracks: Track[];
  target_px: [number, number] | null;
  locked_id: number | null;
  frame_size: [number, number];
  auto_aim: boolean;
  pan: number | null;
  tilt: number | null;
  dmx_type: "mock" | "real";
  beam_px: [number, number] | null;
  calibration: CalibrationStatus;
}

export type ControlMessage =
  | { type: "lock"; track_id: number }
  | { type: "unlock" }
  | { type: "auto_aim"; enabled: boolean }
  | { type: "aim"; pan: number; tilt: number }
  | {
      type: "calibrate_sample";
      pan: number;
      tilt: number;
      px: number | null;
      py: number | null;
    }
  | { type: "calibration_fit" }
  | { type: "calibration_clear" };

export type Mode = "run" | "calibrate";
