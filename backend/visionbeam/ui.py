"""
Director's Station UI (PySide6).

Real-time operator interface connected to the pipeline via display_queue.
Renders the warped top-down floor plan with person-masked motion heatmap
overlay, tracked person bounding boxes with persistent IDs, and current
light aim indicators. Supports click-to-aim manual override and a guided
calibration wizard for ArUco homography and light triangulation setup.
"""
