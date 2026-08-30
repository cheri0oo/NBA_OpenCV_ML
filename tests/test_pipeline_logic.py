import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ball_aquisition import BallAquisitionDetector
from court_mapper import CourtMiniMapDrawer
from drawers import PassInterceptionDrawer
from made_basket_detector import MadeBasketDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from team_assigner import TeamAssigner
from trackers import BallTracker


class PipelineLogicTests(unittest.TestCase):
    def test_short_ball_gaps_are_filled_but_long_gaps_are_not(self):
        positions = [
            {1: {"bbox": [0, 0, 10, 10]}},
            {},
            {1: {"bbox": [20, 0, 30, 10]}},
            {},
            {},
            {},
            {1: {"bbox": [60, 0, 70, 10]}},
        ]
        result = BallTracker.interpolate_ball_positions(positions, max_gap=1)
        self.assertTrue(result[1][1]["interpolated"])
        self.assertEqual(result[3], {})
        self.assertEqual(result[4], {})
        self.assertEqual(result[5], {})

    def test_possession_confirmation_removes_one_frame_flicker(self):
        players = [
            {1: {"bbox": [0, 0, 20, 100]}, 2: {"bbox": [100, 0, 120, 100]}}
            for _ in range(6)
        ]
        ball_tracks = []
        for center in [(10, 50)] * 3 + [(110, 50)] * 3:
            ball_tracks.append({1: {"center": center, "bbox": [center[0]-5, center[1]-5, center[0]+5, center[1]+5], "observed": True}})
        possession = BallAquisitionDetector().detect_ball_possession(players, ball_tracks)
        self.assertEqual(possession, [-1, -1, 1, 1, 1, 2])

    def test_pass_and_interception_event_schema(self):
        possession = [1, 1, -1, 2, 2, 3]
        assignments = [
            {1: 1, 2: 1, 3: 2}
            for _ in possession
        ]
        passes, interceptions = PassAndInterceptionDetector().detect_passes_and_interceptions(
            possession, assignments
        )
        self.assertEqual(passes[0]["team"], 1)
        self.assertEqual(passes[0]["from"], 1)
        self.assertEqual(passes[0]["to"], 2)
        self.assertEqual(interceptions[0]["to_team"], 2)
        self.assertEqual(PassInterceptionDrawer().get_stats(passes, interceptions), (1, 0, 0, 1))

    def test_made_basket_crosses_moving_rim_downward(self):
        balls = [(15, 5), (16, 12), (17, 22)]
        hoops = [[10, 10, 20, 25], [11, 10, 21, 25], [12, 10, 22, 25]]
        events = MadeBasketDetector(min_downward_speed=2).detect_made_baskets(balls, hoops)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "made")

    def test_team_clustering_is_stable_per_track(self):
        frames = []
        tracks = []
        for _ in range(6):
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            frame[10:90, 10:70] = (245, 245, 245)
            frame[10:90, 120:180] = (0, 0, 220)
            frames.append(frame)
            tracks.append({1: {"bbox": [10, 10, 70, 90]}, 2: {"bbox": [120, 10, 180, 90]}})
        with tempfile.TemporaryDirectory() as directory:
            assignments = TeamAssigner(recheck_interval=1, seed_map={2: 2}).get_player_teams_across_frames(
                frames,
                tracks,
                read_from_stub=False,
                stub_path=str(Path(directory) / "teams.pkl"),
            )
        self.assertTrue(all(frame[1] == 1 and frame[2] == 2 for frame in assignments))

    def test_homography_maps_image_corners_to_court_corners(self):
        calibration = {
            "source_points_normalized": False,
            "source_points": [[0, 0], [200, 0], [200, 100], [0, 100]],
            "court_points": [[0, 0], [94, 0], [94, 50], [0, 50]],
        }
        drawer = CourtMiniMapDrawer(calibration=calibration)
        mapped = drawer.map_point((200, 100), 0, (100, 200, 3))
        expected = drawer._court_to_map([[94, 50]])[0]
        self.assertTrue(np.allclose(mapped, expected, atol=0.5))


if __name__ == "__main__":
    unittest.main()
