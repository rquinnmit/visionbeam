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
          <span>Pan</span>
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
          <span>Tilt</span>
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
          Click the image where the beam dot lands to record a sample.
        </span>
        <button onClick={onOffScreen}>Off-screen</button>
        <button onClick={onFit} disabled={status.n_in_frame < 6}>
          Fit · {status.n_in_frame}
        </button>
        <button onClick={onClear}>Clear</button>
      </div>
      <div className="cal-row cal-status">
        <span>
          samples · <strong>{status.n_in_frame}</strong> in /{" "}
          <strong>{status.n_samples}</strong> total
        </span>
        <span>
          fitted ·{" "}
          <strong className={status.fitted ? "ok" : "no"}>
            {status.fitted ? "yes" : "no"}
          </strong>
        </span>
        {status.rms_pan_deg != null && status.rms_tilt_deg != null && (
          <span>
            rms · pan <strong>{status.rms_pan_deg.toFixed(2)}°</strong>, tilt{" "}
            <strong>{status.rms_tilt_deg.toFixed(2)}°</strong>
          </span>
        )}
      </div>
    </div>
  );
}
