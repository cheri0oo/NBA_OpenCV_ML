from .util import draw_elipse, draw_triangle

class PlayerTracksDrawer:
    def __init__(self, team_1_color = (255, 255, 255), team_2_color = (0, 0, 255)):
        # white ring for team 1, blue ring for team 2
        self.default_player_team_id = 0
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self, video_frames, tracks, player_assignment, ball_aquisition):

        output_video_frames = []

        for frame_num, frame_tracks in enumerate(tracks):
            frame = video_frames[frame_num].copy()

            player_dict = tracks[frame_num]
            player_assignment_for_frame = player_assignment[frame_num]
            player_id_has_ball = ball_aquisition[frame_num]

            # Draw Player tracks
            for track_id, player_info in player_dict.items():
                team_id = player_assignment_for_frame.get(
                    track_id,
                    self.default_player_team_id
                )

                # pick correct ring color
                if team_id == 1:
                    color = self.team_1_color
                elif team_id == 2:
                    color = self.team_2_color
                else:
                    color = (160, 160, 160)

                if int(track_id) == int(player_id_has_ball):
                    frame = draw_triangle(frame, player_info["bbox"], (0, 255, 255))


                frame = draw_elipse(
                    frame,
                    player_info['bbox'],
                    color=color,      # use computed team color
                    track_id=track_id
                )

            output_video_frames.append(frame)

        return output_video_frames
