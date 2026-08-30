# NBA OpenCV ML

This project tracks basketball players and the ball, estimates possession, counts passes/interceptions, classifies made baskets, and draws player locations on a top-down mini court.

## Attribution

This project was **heavily inspired by [this YouTube tutorial](https://www.youtube.com/watch?v=QqVahw9tBfw&t=27011s)**.

A substantial portion of the original codebase—approximately half—was written by following or adapting the concepts, structure, and code demonstrated in that tutorial. Full credit goes to the tutorial's creator for the original guidance and foundation.

I expanded the project with additional work including possession smoothing, pass and interception tracking, moving-hoop basket classification, team-assignment improvements, a top-down court minimap, testing, debugging, and pipeline integration.

## Run it

From the `NBA_OpenCV_ML` folder:

```powershell
python -m pip install -r requirements.txt
python main.py
```

The default command processes `input_videos/video_3.mp4` and writes:

```text
output_videos/video_3_output_patched.avi
```

Useful options:

```powershell
# Process a different included clip
python main.py --video input_videos/video_1.mp4

# Ignore cached detections and run both YOLO models again (much slower)
python main.py --fresh

# Hide the mini-map
python main.py --no-mini-map
```

The notebook's final cell calls the same `run_pipeline()` function, so the command line and notebook no longer contain different versions of the pipeline.

## What the overlays mean

- White player marker: Team 1
- Red player marker: Team 2
- Gray player marker: unknown team
- Green ball triangle: observed ball detection
- Orange ball triangle: short predicted/interpolated ball position
- Yellow player triangle/ring: estimated ball holder
- Top-left court: homography-projected player foot positions

Possession changes must remain stable for three frames before a new holder is accepted. Passes are same-team holder changes; interceptions are different-team holder changes. Loose and unknown frames are displayed separately so the three ball-control percentages add to 100%.

## Made baskets

`player_detector.pt` already contains a `Hoop` class. The patched player tracker caches the best moving hoop box in each frame. A made basket is recorded when the tracked ball crosses the detected rim downward inside the hoop's horizontal region.

This is a visual heuristic, not an official scoring system. Occlusion, a missed ball detection, or an inaccurate hoop box can still hide an event. Confirmed event frames can be saved with:

```powershell
python main.py --debug-baskets
```

## Calibrating the mini-map

The mini-map uses a perspective transform (homography), not just resized screen coordinates. It maps the bottom-center of each player bounding box—the player's contact point with the court—onto NBA court coordinates.

Calibration lives in `calibrations/court_calibration.json`. Four `source_points` in the video correspond, in the same order, to four `court_points` measured on a 94-by-50-foot court. Source points are normalized from 0 to 1 by default, so `[0.5, 0.5]` means the center of the video.

The included default produces an approximate mini-map immediately. For accurate absolute court locations, replace the source/court point pairs with known line intersections from your video. A mostly fixed camera needs one four-point calibration. A panning broadcast camera should use `keyframes`; the mapper interpolates the four point sets between those frames:

```json
"keyframes": [
  {
    "frame": 0,
    "source_points": [[0.10, 0.25], [0.90, 0.25], [0.95, 0.90], [0.05, 0.90]],
    "court_points": [[0, 0], [94, 0], [94, 50], [0, 50]]
  },
  {
    "frame": 120,
    "source_points": [[0.08, 0.23], [0.92, 0.23], [0.98, 0.92], [0.02, 0.92]],
    "court_points": [[0, 0], [94, 0], [94, 50], [0, 50]]
  }
]
```

For real broadcast footage, absolute mini-map accuracy ultimately requires reliable court-keypoint detection or manual keyframe calibration because the camera itself pans and zooms.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover limited ball interpolation, possession smoothing, event schemas, team stability, moving-rim basket classification, and the court homography.
