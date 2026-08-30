from __future__ import annotations

import os

import cv2


class MadeBasketDetector:
    """Classify made baskets when the ball crosses a moving rim downward."""

    def __init__(
        self,
        hoop_bbox=None,
        min_downward_speed=2.0,
        persist_frames=1,
        debug_dir=None,
        rim_height_ratio=0.72,
        horizontal_margin=0.35,
        max_observation_gap=5,
        cooldown_frames=30,
    ):
        self.hoop_bbox = hoop_bbox
        self.min_downward_speed = float(min_downward_speed)
        self.persist_frames = persist_frames  # retained for notebook compatibility
        self.debug_dir = debug_dir
        self.rim_height_ratio = rim_height_ratio
        self.horizontal_margin = horizontal_margin
        self.max_observation_gap = max_observation_gap
        self.cooldown_frames = cooldown_frames
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    @staticmethod
    def _ball_center(frame):
        if frame is None:
            return None
        if isinstance(frame, (list, tuple)) and len(frame) == 2:
            return float(frame[0]), float(frame[1])
        if not isinstance(frame, dict):
            return None
        info = frame.get(1, {})
        center = info.get("center")
        if center is not None:
            return float(center[0]), float(center[1])
        bbox = info.get("bbox", [])
        if len(bbox) == 4:
            return (float(bbox[0] + bbox[2]) / 2, float(bbox[1] + bbox[3]) / 2)
        return None

    def _hoop_box(self, frame):
        if frame is None:
            return self.hoop_bbox
        if isinstance(frame, (list, tuple)) and len(frame) == 4:
            return tuple(map(float, frame))
        if isinstance(frame, dict):
            bbox = frame.get(1, {}).get("bbox")
            if bbox is not None and len(bbox) == 4:
                return tuple(map(float, bbox))
        return self.hoop_bbox

    def _rim_geometry(self, bbox):
        x1, y1, x2, y2 = bbox
        width = max(1.0, x2 - x1)
        rim_y = y1 + self.rim_height_ratio * (y2 - y1)
        return (
            x1 - self.horizontal_margin * width,
            x2 + self.horizontal_margin * width,
            rim_y,
        )

    def detect_made_baskets(self, ball_tracks, hoop_tracks=None, frames=None):
        """Return frame-indexed made-basket events.

        ``ball_tracks`` may be center tuples or normal BallTracker dictionaries.
        ``hoop_tracks`` may be moving Hoop dictionaries, constant boxes, or None.
        """
        events = []
        previous = None  # (frame_index, center, rim_y)
        last_event_frame = -10_000

        for frame_index, ball_frame in enumerate(ball_tracks):
            center = self._ball_center(ball_frame)
            hoop_frame = (
                hoop_tracks[frame_index]
                if hoop_tracks is not None and frame_index < len(hoop_tracks)
                else None
            )
            hoop_bbox = self._hoop_box(hoop_frame)
            if center is None or hoop_bbox is None:
                continue

            left, right, rim_y = self._rim_geometry(hoop_bbox)
            if previous is not None:
                previous_frame, previous_center, previous_rim_y = previous
                frame_gap = frame_index - previous_frame
                if frame_gap <= self.max_observation_gap:
                    downward_speed = (center[1] - previous_center[1]) / frame_gap
                    crossed_rim = previous_center[1] <= previous_rim_y and center[1] > rim_y
                    crossing_x = (previous_center[0] + center[0]) / 2
                    close_below_rim = center[1] <= hoop_bbox[3] + (hoop_bbox[3] - hoop_bbox[1])
                    if (
                        crossed_rim
                        and downward_speed >= self.min_downward_speed
                        and left <= crossing_x <= right
                        and close_below_rim
                        and frame_index - last_event_frame >= self.cooldown_frames
                    ):
                        hoop_center = (
                            (hoop_bbox[0] + hoop_bbox[2]) / 2,
                            rim_y,
                        )
                        event = {
                            "frame": int(frame_index),
                            "start_frame": int(previous_frame),
                            "center": hoop_center,
                            "type": "made",
                        }
                        events.append(event)
                        last_event_frame = frame_index
                        if self.debug_dir and frames is not None and frame_index < len(frames):
                            cv2.imwrite(
                                os.path.join(self.debug_dir, f"made_f{frame_index}.jpg"),
                                frames[frame_index],
                            )

            previous = (frame_index, center, rim_y)

        return events

    @staticmethod
    def compute_hoop_bbox_from_reference(frame, approx_center, w=40, h=20):
        x, y = approx_center
        return (
            int(x - w // 2),
            int(y - h // 2),
            int(x + w // 2),
            int(y + h // 2),
        )
