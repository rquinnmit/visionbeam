import type { CalibrationStatus } from "./types";

interface Props {
  pan: number;
  tilt: number;
  panRange: number;
  tiltRange: number;
  status: CalibrationStatus;
  onPanChange: (pan: number) => void;
  onTiltChange: (tilt: number) => void;
  onOffScreen: () => void;
  onFit: () => void;
  onClear: () => void;
}

export function CalibrationPanel({
  pan,
  tilt,
  panRange,
  tiltRange,
  status,
  onPanChange,
  onTiltChange,
  onOffScreen,
  onFit,
  onClear,
}: Props) {
  return (
    <div className="cal-panel">
      <div className="cal-row">
        <label>
          Pan
          <input
            type="range"
            min={0}
            max={panRange}
            step={0.5}
            value={pan}
            onChange={(e) => onPanChange(parseFloat(e.target.value))}
          />
          <span className="cal-val">{pan.toFixed(1)}°</span>
        </label>
      </div>
      <div className="cal-row">
        <label>
          Tilt
          <input
            type="range"
            min={0}
            max={tiltRange}
            step={0.5}
            value={tilt}
            onChange={(e) => onTiltChange(parseFloat(e.target.value))}
          />
          <span className="cal-val">{tilt.toFixed(1)}°</span>
        </label>
      </div>
      <div className="cal-row cal-actions">
        <span className="cal-hint">
          Click image where the beam dot is to record a sample.
        </span>
        <button onClick={onOffScreen}>Off-screen</button>
        <button onClick={onFit} disabled={status.n_in_frame < 6}>
          Fit ({status.n_in_frame})
        </button>
        <button onClick={onClear}>Clear</button>
      </div>
      <div className="cal-row cal-status">
        <span>samples: {status.n_in_frame} in / {status.n_samples} total</span>
        <span>
          fitted:{" "}
          <strong className={status.fitted ? "ok" : "no"}>
            {status.fitted ? "yes" : "no"}
          </strong>
        </span>
        {status.rms_pan_deg != null && status.rms_tilt_deg != null && (
          <span>
            RMS: pan {status.rms_pan_deg.toFixed(2)}°, tilt{" "}
            {status.rms_tilt_deg.toFixed(2)}°
          </span>
        )}
      </div>
    </div>
  );
}
