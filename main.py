from __future__ import annotations

import argparse
import json
from pathlib import Path

from ball_aquisition import BallAquisitionDetector
from court_mapper import CourtMiniMapDrawer
from drawers import (
    BallTracksDrawer,
    MadeBasketsDrawer,
    PassInterceptionDrawer,
    PlayerTracksDrawer,
    TeamBallControlDrawer,
)
from made_basket_detector import MadeBasketDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from team_assigner import TeamAssigner
from trackers import BallTracker, PlayerTracker
from utils import read_video, save_video


PROJECT_ROOT = Path(__file__).resolve().parent


def _merge_dicts(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_video_profile(video_name, calibration_path=None):
    path = Path(calibration_path) if calibration_path else PROJECT_ROOT / "calibrations/court_calibration.json"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)
    return _merge_dicts(profiles.get("default", {}), profiles.get(video_name, {}))


def run_pipeline(
    video_path="input_videos/video_3.mp4",
    output_path=None,
    use_stubs=True,
    calibration_path=None,
    draw_mini_map=True,
    debug_baskets=False,
):
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    video_name = video_path.stem
    output_path = Path(output_path) if output_path else PROJECT_ROOT / f"output_videos/{video_name}_output_patched.avi"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    profile = load_video_profile(video_name, calibration_path)
    video_frames, fps = read_video(str(video_path))

    player_tracker = PlayerTracker(str(PROJECT_ROOT / "models/player_detector.pt"))
    player_tracks, hoop_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=str(PROJECT_ROOT / f"stubs/{video_name}_player_tracks.pkl"),
        hoop_stub_path=str(PROJECT_ROOT / f"stubs/{video_name}_hoop_tracks.pkl"),
        return_hoops=True,
    )

    ball_tracker = BallTracker(str(PROJECT_ROOT / "models/ball_detector_model.pt"))
    ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=str(PROJECT_ROOT / f"stubs/{video_name}_ball_tracks.pkl"),
    )
    ball_tracks = ball_tracker.remove_wrong_detection(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks, max_gap=8)

    team_assigner = TeamAssigner(
        recheck_interval=5,
        seed_map=profile.get("team_seed_map", {}),
    )
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames,
        player_tracks,
        read_from_stub=use_stubs,
        stub_path=str(PROJECT_ROOT / f"stubs/{video_name}_player_assignment.pkl"),
    )

    possession_detector = BallAquisitionDetector()
    possession = possession_detector.detect_ball_possession(player_tracks, ball_tracks)

    event_detector = PassAndInterceptionDetector()
    passes, interceptions = event_detector.detect_passes_and_interceptions(
        possession,
        player_assignment,
    )

    debug_dir = str(PROJECT_ROOT / "debug_made_baskets") if debug_baskets else None
    made_detector = MadeBasketDetector(
        hoop_bbox=profile.get("hoop_bbox"),
        min_downward_speed=2.0,
        debug_dir=debug_dir,
    )
    made_baskets = made_detector.detect_made_baskets(
        ball_tracks,
        hoop_tracks=hoop_tracks,
        frames=video_frames,
    )

    output_frames = PlayerTracksDrawer().draw(
        video_frames,
        player_tracks,
        player_assignment,
        possession,
    )
    output_frames = BallTracksDrawer().draw(output_frames, ball_tracks)
    output_frames = TeamBallControlDrawer().draw(
        output_frames,
        player_assignment,
        possession,
    )
    output_frames = PassInterceptionDrawer().draw(
        output_frames,
        passes,
        interceptions,
    )
    output_frames = MadeBasketsDrawer().draw(output_frames, made_baskets)

    if draw_mini_map:
        output_frames = CourtMiniMapDrawer(profile.get("mini_map", {})).draw(
            output_frames,
            player_tracks,
            player_assignment,
            possession,
        )

    save_video(output_frames, str(output_path), fps)
    summary = {
        "input": str(video_path),
        "output": str(output_path),
        "frames": len(video_frames),
        "fps": fps,
        "passes": passes,
        "interceptions": interceptions,
        "made_baskets": made_baskets,
    }
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Track NBA players, ball possession, events, and court position.")
    parser.add_argument("--video", default="input_videos/video_3.mp4", help="Input video path")
    parser.add_argument("--output", default=None, help="Output AVI path")
    parser.add_argument("--fresh", action="store_true", help="Ignore cached tracking stubs")
    parser.add_argument("--no-mini-map", action="store_true", help="Disable the top-down court")
    parser.add_argument("--calibration", default=None, help="Court calibration JSON path")
    parser.add_argument("--debug-baskets", action="store_true", help="Save confirmed basket frames")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        use_stubs=not args.fresh,
        calibration_path=args.calibration,
        draw_mini_map=not args.no_mini_map,
        debug_baskets=args.debug_baskets,
    )
