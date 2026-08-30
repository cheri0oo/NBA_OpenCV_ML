from __future__ import annotations

import numpy as np
import supervision as sv
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

from utils import read_stubs, save_stubs


class KalmanBallTracker:
    """Small constant-velocity Kalman filter for the ball center."""

    def __init__(self, max_jump=140):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        self.kf.R *= 15.0
        self.kf.P *= 10.0
        self.kf.Q *= 0.01
        self.initialized = False
        self.last_detection = None
        self.max_jump = max_jump

    def update(self, detection):
        if not self.initialized:
            if detection is None:
                return None
            x, y = map(float, detection)
            self.kf.x = np.array([[x], [y], [0.0], [0.0]])
            self.last_detection = (x, y)
            self.initialized = True
            return x, y

        self.kf.predict()
        if detection is not None:
            detection = tuple(map(float, detection))
            if self.last_detection is not None:
                jump = np.linalg.norm(np.asarray(detection) - np.asarray(self.last_detection))
                if jump > self.max_jump:
                    detection = None

        if detection is not None:
            self.kf.update(np.asarray(detection))
            self.last_detection = detection

        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])


class BallTracker:
    def __init__(self, model_path, max_prediction_gap=4):
        self.model = YOLO(model_path)
        self.kalman = KalmanBallTracker()
        self.max_prediction_gap = max_prediction_gap
        self.missed_frames = 0

    def detect_ball(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            detections.extend(self.model.predict(batch, conf=0.5, verbose=False))
        return detections

    @staticmethod
    def _ball_class_id(names):
        for class_id, name in names.items():
            if str(name).lower() == "ball":
                return int(class_id)
        return None

    def _choose_detection(self, detection):
        ball_class_id = self._ball_class_id(detection.names)
        if ball_class_id is None:
            return None

        converted = sv.Detections.from_ultralytics(detection)
        candidates = []
        for item in converted:
            bbox, confidence, class_id = item[0].tolist(), float(item[2]), int(item[3])
            if class_id != ball_class_id:
                continue
            x1, y1, x2, y2 = bbox
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if not 20 <= area <= 3000:
                continue
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
            if self.kalman.last_detection is None:
                continuity_distance = 0.0
            else:
                continuity_distance = float(
                    np.linalg.norm(center - np.asarray(self.kalman.last_detection))
                )
            candidates.append((continuity_distance, -confidence, bbox, confidence, center))

        if not candidates:
            return None

        if self.kalman.last_detection is None:
            candidates.sort(key=lambda item: item[1])
        else:
            nearby = [item for item in candidates if item[0] <= self.kalman.max_jump]
            candidates = nearby or candidates
            candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0]

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        tracks = read_stubs(read_from_stub, stub_path)
        if isinstance(tracks, list) and len(tracks) == len(frames):
            return tracks

        detections = self.detect_ball(frames)
        tracks = []
        for detection in detections:
            selected = self._choose_detection(detection)
            if selected is not None:
                _, _, bbox, confidence, center = selected
                filtered = self.kalman.update(center)
                self.missed_frames = 0
                observed = True
                width = max(8.0, min(30.0, bbox[2] - bbox[0]))
                height = max(8.0, min(30.0, bbox[3] - bbox[1]))
            else:
                filtered = self.kalman.update(None)
                self.missed_frames += 1
                observed = False
                confidence = None
                width = height = 10.0

            if filtered is None or self.missed_frames > self.max_prediction_gap:
                tracks.append({})
                continue

            x, y = filtered
            tracks.append({
                1: {
                    "bbox": [x - width / 2, y - height / 2, x + width / 2, y + height / 2],
                    "center": [x, y],
                    "confidence": confidence,
                    "observed": observed,
                }
            })

        save_stubs(stub_path, tracks)
        return tracks

    @staticmethod
    def remove_wrong_detection(ball_positions, maximum_speed=35):
        """Drop implausible jumps while allowing more motion across missed frames."""
        output = []
        last_center = None
        last_good_index = None
        for index, frame in enumerate(ball_positions):
            info = frame.get(1, {}) if isinstance(frame, dict) else {}
            bbox = info.get("bbox", [])
            if len(bbox) != 4:
                output.append({})
                continue
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            if last_center is not None:
                frame_gap = index - last_good_index
                if np.linalg.norm(center - last_center) > maximum_speed * frame_gap:
                    output.append({})
                    continue
            copied = dict(info)
            copied["bbox"] = list(map(float, bbox))
            copied["center"] = center.tolist()
            copied.setdefault("observed", True)
            output.append({1: copied})
            last_center = center
            last_good_index = index
        return output

    @staticmethod
    def interpolate_ball_positions(ball_positions, max_gap=8):
        """Linearly fill only short internal gaps; long/edge gaps stay unknown."""
        output = []
        for frame in ball_positions:
            info = frame.get(1, {}) if isinstance(frame, dict) else {}
            bbox = info.get("bbox", [])
            if len(bbox) == 4:
                copied = dict(info)
                copied["bbox"] = list(map(float, bbox))
                copied["center"] = [
                    (copied["bbox"][0] + copied["bbox"][2]) / 2,
                    (copied["bbox"][1] + copied["bbox"][3]) / 2,
                ]
                copied.setdefault("observed", True)
                output.append({1: copied})
            else:
                output.append({})

        known = [i for i, frame in enumerate(output) if frame.get(1)]
        for left, right in zip(known, known[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > max_gap:
                continue
            left_box = np.asarray(output[left][1]["bbox"], dtype=float)
            right_box = np.asarray(output[right][1]["bbox"], dtype=float)
            for offset in range(1, gap + 1):
                ratio = offset / (gap + 1)
                bbox = ((1.0 - ratio) * left_box + ratio * right_box).tolist()
                output[left + offset] = {
                    1: {
                        "bbox": bbox,
                        "center": [
                            (bbox[0] + bbox[2]) / 2,
                            (bbox[1] + bbox[3]) / 2,
                        ],
                        "confidence": None,
                        "observed": False,
                        "interpolated": True,
                    }
                }
        return output

    @staticmethod
    def get_ball_centers(ball_tracks):
        centers = []
        for frame in ball_tracks:
            info = frame.get(1, {}) if isinstance(frame, dict) else {}
            center = info.get("center")
            if center is None:
                bbox = info.get("bbox", [])
                if len(bbox) == 4:
                    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            centers.append(tuple(map(float, center)) if center is not None else None)
        return centers
