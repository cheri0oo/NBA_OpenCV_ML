import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width


def draw_triangle(frame, bbox, color):
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20]
    ])

    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
    return frame

def draw_elipse(frame, bbox, color, track_id=None):
    # Sanitize bbox
    if (
        bbox is None or
        len(bbox) != 4 or
        any(b is None for b in bbox)
    ):
        return frame

    try:
        x1, y1, x2, y2 = map(int, bbox)
    except:
        return frame

    # Center
    x_center, y_center = get_center_of_bbox((x1, y1, x2, y2))

    # Better ellipse size
    width = max(1, get_bbox_width((x1, y1, x2, y2)))
    axes = (int(width * 1.2), int(width * 0.5))   # <-- bigger + more oval

    # Draw ellipse
    cv2.ellipse(
        frame,
        (x_center, y2),
        axes,
        0,
        -45,
        235,
        color,
        2,
        cv2.LINE_4
    )

    # Draw ID box
    rectangle_width = 40
    rectangle_height = 20

    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = y2 - rectangle_height // 2
    y2_rect = y2 + rectangle_height // 2

    if track_id is not None:
        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
        cv2.putText(frame, str(track_id), (x_center - 8, y2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame


