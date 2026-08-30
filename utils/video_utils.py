import cv2
import os

# read video frames from a video file
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)  # get original FPS

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:  # last frame reached
            break
        frames.append(frame)

    cap.release()
    return frames, fps


# saving video 
def save_video(output_video_frames, output_video_path, fps):
    if not output_video_frames:
        raise ValueError("Cannot save a video with zero frames")

    # delete old file so OpenCV doesn't corrupt the new one
    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    # create directory if missing
    directory = os.path.dirname(output_video_path)
    if directory != "" and not os.path.exists(directory):
        os.mkdir(directory)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    height = output_video_frames[0].shape[0]
    width = output_video_frames[0].shape[1]

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    for frame in output_video_frames:
        out.write(frame)

    out.release()
