from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from utils import read_stubs, save_stubs


class TeamAssigner:
    """Assign stable team IDs by clustering each track's jersey colors."""

    CACHE_VERSION = 2

    def __init__(self, recheck_interval=5, persistence=5, seed_map=None):
        self.recheck_interval = max(1, int(recheck_interval))
        self.persistence = persistence  # kept for compatibility with older notebooks
        self.player_to_team = {
            int(player_id): int(team_id)
            for player_id, team_id in (seed_map or {}).items()
        }

    @staticmethod
    def _jersey_crop(frame, bbox):
        x1, y1, x2, y2 = map(float, bbox)
        height, width = frame.shape[:2]
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)

        # Focus on the upper torso: avoid court, shorts, head, and most skin.
        left = int(max(0, x1 + 0.18 * box_width))
        right = int(min(width, x2 - 0.18 * box_width))
        top = int(max(0, y1 + 0.20 * box_height))
        bottom = int(min(height, y1 + 0.58 * box_height))
        return frame[top:bottom, left:right]

    def _jersey_feature(self, frame, bbox):
        crop = self._jersey_crop(frame, bbox)
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None

        crop = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        # Remove nearly black pixels (numbers/shadows) before taking a robust median.
        mask = hsv[..., 2] > 45
        pixels = lab[mask]
        if len(pixels) < 20:
            return None
        return np.median(pixels, axis=0).astype(np.float32)

    def classify_hsv(self, frame, bbox, hist_window=5):
        """Backward-compatible helper returning raw average HSV values."""
        crop = self._jersey_crop(frame, bbox)
        if crop.size == 0:
            return 0.0, 0.0, 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return tuple(float(hsv[..., channel].mean()) for channel in range(3))

    def _cluster_tracks(self, video_frames, player_tracks):
        samples = defaultdict(list)
        for frame_index in range(0, min(len(video_frames), len(player_tracks)), self.recheck_interval):
            frame = video_frames[frame_index]
            for player_id, track in player_tracks[frame_index].items():
                bbox = track.get("bbox", [])
                if len(bbox) != 4:
                    continue
                feature = self._jersey_feature(frame, bbox)
                if feature is not None:
                    samples[int(player_id)].append(feature)

        track_ids = []
        features = []
        for player_id, observations in samples.items():
            track_ids.append(player_id)
            features.append(np.median(np.asarray(observations), axis=0))

        if len(features) < 2:
            return {}

        data = np.asarray(features, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(
            data,
            2,
            None,
            criteria,
            20,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.reshape(-1)

        # Use supplied seed IDs when possible. Otherwise call the brighter jersey
        # Team 1 and the darker/more saturated jersey Team 2.
        seed_votes = {0: defaultdict(int), 1: defaultdict(int)}
        for player_id, label in zip(track_ids, labels):
            if player_id in self.player_to_team:
                seed_votes[int(label)][self.player_to_team[player_id]] += 1

        cluster_to_team = {}
        for cluster_id, votes in seed_votes.items():
            if votes:
                cluster_to_team[cluster_id] = max(votes, key=votes.get)

        if len(set(cluster_to_team.values())) < len(cluster_to_team):
            cluster_to_team = {}
        if not cluster_to_team:
            light_cluster = int(np.argmax(centers[:, 0]))  # L channel in Lab
            cluster_to_team = {light_cluster: 1, 1 - light_cluster: 2}
        elif len(cluster_to_team) == 1:
            known_cluster, known_team = next(iter(cluster_to_team.items()))
            cluster_to_team[1 - known_cluster] = 1 if known_team == 2 else 2

        return {
            player_id: int(cluster_to_team[int(label)])
            for player_id, label in zip(track_ids, labels)
        }

    @staticmethod
    def _normalize(assignments):
        return [
            {int(player_id): int(team_id) for player_id, team_id in frame.items()}
            for frame in assignments
        ]

    def get_player_teams_across_frames(
        self,
        video_frames,
        player_tracks,
        read_from_stub=False,
        stub_path=None,
    ):
        cached = read_stubs(read_from_stub, stub_path)
        if (
            isinstance(cached, dict)
            and cached.get("version") == self.CACHE_VERSION
            and isinstance(cached.get("assignments"), list)
            and len(cached["assignments"]) == len(video_frames)
        ):
            return self._normalize(cached["assignments"])

        seeds = dict(self.player_to_team)
        clustered = self._cluster_tracks(video_frames, player_tracks)
        self.player_to_team.update(clustered)
        self.player_to_team.update(seeds)  # explicit seeds always win

        assignments = []
        for frame_players in player_tracks:
            assignments.append({
                int(player_id): int(self.player_to_team.get(int(player_id), 0))
                for player_id in frame_players
            })

        save_stubs(stub_path, {
            "version": self.CACHE_VERSION,
            "assignments": assignments,
        })
        return assignments
