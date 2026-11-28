import cv2
import time


def make_zone_labels(rows: int, cols: int) -> list[str]:
    """Return Q1..Q4 for standard 2x2 grids, otherwise Z1..ZN."""
    if rows == 2 and cols == 2:
        return ["Q1", "Q2", "Q3", "Q4"]
    return [f"Z{i}" for i in range(1, rows * cols + 1)]


def compute_row_breaks(rows: int, angle_mode: str) -> list[float]:
    """Normalized vertical split points [0.0 .. 1.0] adapted to camera angle."""
    exp = 1.32 if angle_mode in ("elevated_front", "corner_left", "corner_right") else 1.0
    return [round((i / rows) ** exp, 4) for i in range(rows + 1)]


def compute_col_x_norm(col_idx: int, cols: int, y_norm: float, angle_mode: str) -> float:
    """Normalized horizontal boundary x at normalized depth y_norm for camera angle."""
    base = col_idx / cols
    if angle_mode == "elevated_front":
        scale = 0.78 + 0.22 * y_norm
        return max(0.0, min(1.0, 0.5 + (base - 0.5) * scale))
    if angle_mode == "corner_left":
        shift = 0.10 - 0.20 * y_norm
        return max(0.0, min(1.0, base + shift)) if 0 < col_idx < cols else base
    if angle_mode == "corner_right":
        shift = -0.10 + 0.20 * y_norm
        return max(0.0, min(1.0, base + shift)) if 0 < col_idx < cols else base
    return base


class Visualizer:
    def __init__(self):
        self.prev_time = 0

    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def draw_zones(self, frame, counts: dict, rows: int = 2, cols: int = 2, angle_mode: str = "overhead"):
        """Overlay dynamic N-zone grid boundaries adapted to camera perspective."""
        h, w = frame.shape[:2]
        line_color = (255, 200, 0)
        text_color = (255, 220, 80)
        font = cv2.FONT_HERSHEY_SIMPLEX

        y_breaks = compute_row_breaks(rows, angle_mode)

        # Horizontal row dividers
        for r in range(1, rows):
            py = int(y_breaks[r] * h)
            cv2.line(frame, (0, py), (w, py), line_color, 1, cv2.LINE_AA)

        # Vertical / perspective column dividers
        for c in range(1, cols):
            x_top = int(compute_col_x_norm(c, cols, 0.0, angle_mode) * w)
            x_bot = int(compute_col_x_norm(c, cols, 1.0, angle_mode) * w)
            cv2.line(frame, (x_top, 0), (x_bot, h), line_color, 1, cv2.LINE_AA)

        labels = make_zone_labels(rows, cols)
        for r in range(rows):
            ym = 0.5 * (y_breaks[r] + y_breaks[r + 1])
            cy = int(ym * h)
            for c in range(cols):
                idx = r * cols + c
                lbl = labels[idx]
                x_left = compute_col_x_norm(c, cols, ym, angle_mode)
                x_right = compute_col_x_norm(c + 1, cols, ym, angle_mode)
                tx = max(8, int(((x_left + x_right) * 0.5) * w) - 38)
                ty = max(24, int(y_breaks[r] * h) + 26)
                cv2.putText(frame, f"{lbl}: {counts.get(lbl, 0)}", (tx, ty),
                            font, 0.62, text_color, 2, cv2.LINE_AA)

        total = sum(counts.values())
        cv2.putText(frame, f"Total persons: {total} ({rows}x{cols} {angle_mode})",
                    (10, h - 14), font, 0.62, (0, 200, 255), 2, cv2.LINE_AA)
        return frame

    def draw_quadrants(self, frame, counts):
        """Backward-compatible wrapper for default 2x2 grid."""
        return self.draw_zones(frame, counts, rows=2, cols=2, angle_mode="overhead")

    def draw_fps(self, frame):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame
