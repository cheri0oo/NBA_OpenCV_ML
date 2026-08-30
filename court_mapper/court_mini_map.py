from __future__ import annotations

import cv2
import numpy as np

from utils import get_foot_position


class CourtMiniMapDrawer:
    """Project player foot positions onto a top-down NBA court with a homography."""

    COURT_LENGTH = 94.0
    COURT_WIDTH = 50.0

    def __init__(self, calibration=None, map_width=320, margin=14, alpha=0.90):
        self.calibration = calibration or {}
        self.map_width = int(self.calibration.get("map_width", map_width))
        self.court_padding = 12
        usable_width = self.map_width - 2 * self.court_padding
        usable_height = usable_width * self.COURT_WIDTH / self.COURT_LENGTH
        self.map_height = int(round(usable_height + 2 * self.court_padding + 24))
        self.margin = int(self.calibration.get("margin", margin))
        self.alpha = float(self.calibration.get("alpha", alpha))
        self.position = tuple(self.calibration.get("position", [self.margin, self.margin]))
        self.show_track_ids = bool(self.calibration.get("show_track_ids", False))

    def _court_to_map(self, points):
        points = np.asarray(points, dtype=np.float32)
        width = self.map_width - 2 * self.court_padding
        height = self.map_height - 24 - 2 * self.court_padding
        mapped = np.empty_like(points)
        mapped[:, 0] = self.court_padding + points[:, 0] / self.COURT_LENGTH * width
        mapped[:, 1] = 24 + self.court_padding + points[:, 1] / self.COURT_WIDTH * height
        return mapped

    @staticmethod
    def _interpolate_points(left, right, ratio):
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        return (1.0 - ratio) * left + ratio * right

    def _calibration_for_frame(self, frame_index, frame_shape):
        config = self.calibration
        keyframes = sorted(config.get("keyframes", []), key=lambda item: int(item["frame"]))
        if keyframes:
            if frame_index <= int(keyframes[0]["frame"]):
                source = keyframes[0]["source_points"]
                court = keyframes[0].get("court_points", config.get("court_points"))
            elif frame_index >= int(keyframes[-1]["frame"]):
                source = keyframes[-1]["source_points"]
                court = keyframes[-1].get("court_points", config.get("court_points"))
            else:
                for left, right in zip(keyframes, keyframes[1:]):
                    left_frame, right_frame = int(left["frame"]), int(right["frame"])
                    if left_frame <= frame_index <= right_frame:
                        ratio = (frame_index - left_frame) / max(1, right_frame - left_frame)
                        source = self._interpolate_points(
                            left["source_points"], right["source_points"], ratio
                        )
                        left_court = left.get("court_points", config.get("court_points"))
                        right_court = right.get("court_points", config.get("court_points"))
                        court = self._interpolate_points(left_court, right_court, ratio)
                        break
        else:
            source = config.get("source_points", [
                [0.03, 0.24],
                [0.97, 0.24],
                [0.98, 0.96],
                [0.02, 0.96],
            ])
            court = config.get("court_points", [
                [0.0, 0.0],
                [94.0, 0.0],
                [94.0, 50.0],
                [0.0, 50.0],
            ])

        source = np.asarray(source, dtype=np.float32)
        normalized = bool(config.get("source_points_normalized", True))
        if normalized:
            height, width = frame_shape[:2]
            source = source * np.asarray([width, height], dtype=np.float32)
        return source, np.asarray(court, dtype=np.float32)

    def _homography(self, frame_index, frame_shape):
        source, court = self._calibration_for_frame(frame_index, frame_shape)
        destination = self._court_to_map(court)
        homography, _ = cv2.findHomography(source, destination, method=0)
        return homography

    def map_point(self, point, frame_index, frame_shape):
        homography = self._homography(frame_index, frame_shape)
        if homography is None:
            return None
        point_array = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point_array, homography)[0, 0]
        if not np.isfinite(mapped).all():
            return None
        return float(mapped[0]), float(mapped[1])

    def _draw_court(self):
        canvas = np.full((self.map_height, self.map_width, 3), (52, 94, 128), dtype=np.uint8)
        cv2.putText(
            canvas,
            "TOP-DOWN COURT",
            (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        corners = self._court_to_map([[0, 0], [94, 0], [94, 50], [0, 50]]).astype(int)
        cv2.polylines(canvas, [corners], True, (255, 255, 255), 2)

        half_top, half_bottom = self._court_to_map([[47, 0], [47, 50]]).astype(int)
        cv2.line(canvas, tuple(half_top), tuple(half_bottom), (255, 255, 255), 1)
        center = self._court_to_map([[47, 25]])[0].astype(int)
        radius = max(5, int((self.map_width - 2 * self.court_padding) * 6 / 94))
        cv2.circle(canvas, tuple(center), radius, (255, 255, 255), 1)

        for basket_x, direction in ((5.25, 1), (88.75, -1)):
            basket = self._court_to_map([[basket_x, 25]])[0].astype(int)
            cv2.circle(canvas, tuple(basket), 2, (255, 255, 255), cv2.FILLED)
            lane_x = basket_x + direction * 13.75
            lane = self._court_to_map([
                [basket_x, 17], [lane_x, 17], [lane_x, 33], [basket_x, 33]
            ]).astype(int)
            cv2.polylines(canvas, [lane], True, (255, 255, 255), 1)
        return canvas

    def draw_frame(self, frame, frame_index, player_tracks, player_assignment, ball_holder=-1):
        mini_map = self._draw_court()
        for player_id, track in player_tracks.items():
            bbox = track.get("bbox", [])
            if len(bbox) != 4:
                continue
            mapped = self.map_point(get_foot_position(bbox), frame_index, frame.shape)
            if mapped is None:
                continue
            x, y = map(int, mapped)
            if not (self.court_padding - 5 <= x < self.map_width - self.court_padding + 5):
                continue
            if not (24 + self.court_padding - 5 <= y < self.map_height - self.court_padding + 5):
                continue

            team_id = int(player_assignment.get(int(player_id), 0))
            color = (245, 245, 245) if team_id == 1 else (40, 40, 230) if team_id == 2 else (150, 150, 150)
            cv2.circle(mini_map, (x, y), 6, (0, 0, 0), cv2.FILLED)
            cv2.circle(mini_map, (x, y), 4, color, cv2.FILLED)
            if int(player_id) == int(ball_holder):
                cv2.circle(mini_map, (x, y), 8, (0, 255, 255), 2)
            if self.show_track_ids:
                cv2.putText(
                    mini_map,
                    str(player_id),
                    (x + 6, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        x0, y0 = map(int, self.position)
        available_width = frame.shape[1] - x0
        available_height = frame.shape[0] - y0
        if available_width <= 0 or available_height <= 0:
            return frame
        mini_map = mini_map[:available_height, :available_width]
        region = frame[y0:y0 + mini_map.shape[0], x0:x0 + mini_map.shape[1]]
        cv2.addWeighted(mini_map, self.alpha, region, 1.0 - self.alpha, 0, region)
        return frame

    def draw(self, video_frames, player_tracks, player_assignment, ball_aquisition):
        for frame_index, frame in enumerate(video_frames):
            self.draw_frame(
                frame,
                frame_index,
                player_tracks[frame_index] if frame_index < len(player_tracks) else {},
                player_assignment[frame_index] if frame_index < len(player_assignment) else {},
                ball_aquisition[frame_index] if frame_index < len(ball_aquisition) else -1,
            )
        return video_frames
