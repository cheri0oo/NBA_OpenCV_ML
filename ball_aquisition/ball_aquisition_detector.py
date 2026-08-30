from __future__ import annotations

from utils.bbox_utils import get_center_of_bbox, measure_distance


class BallAquisitionDetector:
    """Estimate a stable ball holder from ball and player tracks."""

    def __init__(
        self,
        close_distance=40,
        air_pass_distance=75,
        switch_confirmation_frames=3,
        max_hold_gap_frames=2,
    ):
        self.close_distance = close_distance
        self.air_pass_distance = air_pass_distance
        self.switch_confirmation_frames = max(1, int(switch_confirmation_frames))
        self.max_hold_gap_frames = max(0, int(max_hold_gap_frames))

    @staticmethod
    def get_key_basketball_player_assignment_points(player_bbox, ball_center=None):
        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1
        return [
            (x1 + width * 0.20, y1 + height * 0.55),  # left hand
            (x2 - width * 0.20, y1 + height * 0.55),  # right hand
            (x1 + width * 0.30, y1 + height * 0.45),  # left elbow
            (x2 - width * 0.30, y1 + height * 0.45),  # right elbow
            (x1 + width * 0.50, y1 + height * 0.15),  # head
            (x1 + width * 0.50, y1 + height * 0.35),  # upper torso
            (x1 + width * 0.50, y1 + height * 0.50),  # mid torso
        ]

    def find_nearest_player(self, ball_center, player_tracks_frame):
        best_player = -1
        best_distance = float("inf")
        for player_id, info in player_tracks_frame.items():
            bbox = info.get("bbox", [])
            if len(bbox) != 4:
                continue
            points = self.get_key_basketball_player_assignment_points(bbox, ball_center)
            distance = min(measure_distance(ball_center, point) for point in points)
            if distance < best_distance:
                best_distance = distance
                best_player = int(player_id)
        return best_player, best_distance

    @staticmethod
    def _ball_center_and_observed(frame):
        info = frame.get(1, {}) if isinstance(frame, dict) else {}
        center = info.get("center")
        if center is None:
            bbox = info.get("bbox", [])
            if len(bbox) == 4:
                center = get_center_of_bbox(bbox)
        return center, bool(info.get("observed", True)) if center is not None else False

    def detect_ball_possession(self, player_tracks, ball_tracks):
        possession = []
        holder = -1
        switch_candidate = -1
        switch_count = 0
        loose_count = 0

        for frame_index, ball_frame in enumerate(ball_tracks):
            center, observed = self._ball_center_and_observed(ball_frame)
            frame_players = player_tracks[frame_index] if frame_index < len(player_tracks) else {}
            if center is None or not frame_players:
                loose_count += 1
                if loose_count > self.max_hold_gap_frames:
                    holder = -1
                possession.append(-1)
                continue

            nearest_player, distance = self.find_nearest_player(center, frame_players)
            if nearest_player == -1 or distance > self.air_pass_distance:
                loose_count += 1
                switch_candidate = -1
                switch_count = 0
                if loose_count > self.max_hold_gap_frames:
                    holder = -1
                possession.append(-1)
                continue

            if distance >= self.close_distance:
                loose_count += 1
                possession.append(holder if holder != -1 and loose_count <= self.max_hold_gap_frames else -1)
                continue

            loose_count = 0
            if nearest_player == holder:
                switch_candidate = -1
                switch_count = 0
                possession.append(holder)
                continue

            # Interpolated/predicted ball locations may maintain a holder but are
            # not trusted to create a new possession change.
            if not observed:
                possession.append(holder)
                continue

            if nearest_player == switch_candidate:
                switch_count += 1
            else:
                switch_candidate = nearest_player
                switch_count = 1

            if switch_count >= self.switch_confirmation_frames:
                holder = switch_candidate
                switch_candidate = -1
                switch_count = 0

            possession.append(holder)

        return [int(player_id) for player_id in possession]
