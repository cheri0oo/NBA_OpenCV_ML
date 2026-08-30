import cv2

class PassInterceptionDrawer:

    def __init__(self):
        pass

    def get_stats(self, passes, interceptions):
        def pass_team(event):
            return event.get("team", 0) if isinstance(event, dict) else event

        def interception_team(event):
            return event.get("to_team", 0) if isinstance(event, dict) else event

        team_1_passes = sum(1 for event in passes if pass_team(event) == 1)
        team_2_passes = sum(1 for event in passes if pass_team(event) == 2)
        team_1_interceptions = sum(1 for event in interceptions if interception_team(event) == 1)
        team_2_interceptions = sum(1 for event in interceptions if interception_team(event) == 2)

        return (
            team_1_passes,
            team_2_passes,
            team_1_interceptions,
            team_2_interceptions
        )

    def draw(self, video_frames, passes, interceptions):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame_drawn = self.draw_frame(
                frame,
                frame_num,
                passes,
                interceptions
            )
            output_video_frames.append(frame_drawn)

        return output_video_frames

    def draw_frame(self, frame, frame_num, passes, interceptions):

        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2

        frame_height, frame_width, _ = overlay.shape

        # === Overlay rectangle ===
        rect_x1 = int(frame_width * 0.16)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.55)
        rect_y2 = int(frame_height * 0.90)

        cv2.rectangle(
            overlay,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            (255, 255, 255),
            -1
        )

        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # === Stats ===
        passes_till_frame = [
            event for event in passes
            if not isinstance(event, dict) or int(event.get("frame", 0)) <= frame_num
        ]
        interceptions_till_frame = [
            event for event in interceptions
            if not isinstance(event, dict) or int(event.get("frame", 0)) <= frame_num
        ]

        (
            team_1_passes,
            team_2_passes,
            team_1_interceptions,
            team_2_interceptions
        ) = self.get_stats(passes_till_frame, interceptions_till_frame)

        # === Text positions anchored to rectangle ===
        text_x = rect_x1 + 20
        text_y1 = rect_y1 + 35   # Team 1 (top)
        text_y2 = rect_y1 + 75   # Team 2 (bottom)

        # === Draw text ===
        cv2.putText(
            frame,
            f"Team 1 - Passes: {team_1_passes}  Interceptions: {team_1_interceptions}",
            (text_x, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness
        )

        cv2.putText(
            frame,
            f"Team 2 - Passes: {team_2_passes}  Interceptions: {team_2_interceptions}",
            (text_x, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness
        )

        return frame
