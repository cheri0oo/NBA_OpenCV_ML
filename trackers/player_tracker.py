from __future__ import annotations

import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack
from ultralytics import YOLO

from utils.stubs_util import read_stubs, save_stubs


class PlayerTracker:
    """Detect and track players, while also caching the moving hoop box."""

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = ByteTrack()

    def detect_players(self, frames):
        """Run the multi-class model in small batches.

        The historical method name is kept so existing notebook imports continue
        to work. The returned YOLO results also contain Hoop detections.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            detections.extend(self.model.predict(batch, conf=0.5, verbose=False))
        return detections

    @staticmethod
    def _class_id(names, wanted_name):
        wanted_name = wanted_name.lower()
        for class_id, name in names.items():
            if str(name).lower() == wanted_name:
                return int(class_id)
        return None

    def _build_tracks(self, detections):
        player_tracks = []
        hoop_tracks = []

        for detection in detections:
            names = detection.names
            converted = sv.Detections.from_ultralytics(detection)
            player_class_id = self._class_id(names, "Player")
            hoop_class_id = self._class_id(names, "Hoop")

            frame_players = {}
            if player_class_id is not None and converted.class_id is not None:
                player_detections = converted[converted.class_id == player_class_id]
                tracked = self.tracker.update_with_detections(player_detections)
                for item in tracked:
                    bbox, confidence, class_id, track_id = item[0], item[2], item[3], item[4]
                    if track_id is None:
                        continue
                    frame_players[int(track_id)] = {
                        "bbox": bbox.tolist(),
                        "confidence": float(confidence) if confidence is not None else None,
                        "class_id": int(class_id),
                        "class_name": names[int(class_id)],
                    }
            else:
                self.tracker.update_with_detections(sv.Detections.empty())
            player_tracks.append(frame_players)

            best_hoop = None
            best_confidence = -1.0
            if hoop_class_id is not None and converted.class_id is not None:
                for item in converted[converted.class_id == hoop_class_id]:
                    confidence = float(item[2]) if item[2] is not None else 0.0
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_hoop = item[0].tolist()

            if best_hoop is None:
                hoop_tracks.append({})
            else:
                hoop_tracks.append({
                    1: {
                        "bbox": best_hoop,
                        "confidence": best_confidence,
                        "observed": True,
                    }
                })

        return player_tracks, self.interpolate_hoop_positions(hoop_tracks)

    @staticmethod
    def interpolate_hoop_positions(hoop_tracks, max_gap=12):
        """Fill short missed hoop detections without inventing long trajectories."""
        output = [dict(frame) for frame in hoop_tracks]
        known = [i for i, frame in enumerate(hoop_tracks) if frame.get(1, {}).get("bbox")]

        for left, right in zip(known, known[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > max_gap:
                continue
            left_box = np.asarray(hoop_tracks[left][1]["bbox"], dtype=float)
            right_box = np.asarray(hoop_tracks[right][1]["bbox"], dtype=float)
            for offset in range(1, gap + 1):
                ratio = offset / (gap + 1)
                bbox = ((1.0 - ratio) * left_box + ratio * right_box).tolist()
                output[left + offset] = {1: {"bbox": bbox, "observed": False}}

        return output

    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None,
        hoop_stub_path=None,
        return_hoops=False,
    ):
        player_cache = read_stubs(read_from_stub, stub_path)
        hoop_cache = read_stubs(read_from_stub, hoop_stub_path)
        players_valid = isinstance(player_cache, list) and len(player_cache) == len(frames)
        hoops_valid = isinstance(hoop_cache, list) and len(hoop_cache) == len(frames)

        if players_valid and (not return_hoops or hoops_valid):
            if return_hoops:
                return player_cache, hoop_cache
            return player_cache

        detections = self.detect_players(frames)
        detected_players, detected_hoops = self._build_tracks(detections)

        player_tracks = player_cache if players_valid else detected_players
        hoop_tracks = hoop_cache if hoops_valid else detected_hoops
        if not players_valid:
            save_stubs(stub_path, player_tracks)
        if return_hoops and not hoops_valid:
            save_stubs(hoop_stub_path, hoop_tracks)

        if return_hoops:
            return player_tracks, hoop_tracks
        return player_tracks
