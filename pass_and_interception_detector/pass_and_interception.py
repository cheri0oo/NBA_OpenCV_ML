class PassAndInterceptionDetector:
    """Turn stable possession changes into pass or interception events."""

    def __init__(self, max_transition_gap_frames=45, duplicate_cooldown_frames=8):
        self.max_transition_gap_frames = max_transition_gap_frames
        self.duplicate_cooldown_frames = duplicate_cooldown_frames

    def detect_passes_and_interceptions(self, possession_list, team_assignment_per_frame):
        passes = []
        interceptions = []
        previous_holder = -1
        previous_team = -1
        previous_seen_frame = -1
        last_event_frame = {}

        for frame_index, holder in enumerate(possession_list):
            holder = int(holder)
            if holder == -1 or frame_index >= len(team_assignment_per_frame):
                continue

            current_team = int(team_assignment_per_frame[frame_index].get(holder, 0))
            if current_team not in (1, 2):
                continue

            if previous_holder == -1:
                previous_holder = holder
                previous_team = current_team
                previous_seen_frame = frame_index
                continue

            if holder == previous_holder:
                previous_team = current_team
                previous_seen_frame = frame_index
                continue

            gap = frame_index - previous_seen_frame
            if gap <= self.max_transition_gap_frames:
                event_key = (previous_holder, holder, previous_team, current_team)
                last_frame = last_event_frame.get(event_key, -10_000)
                if frame_index - last_frame >= self.duplicate_cooldown_frames:
                    if current_team == previous_team:
                        passes.append({
                            "frame": int(frame_index),
                            "from": int(previous_holder),
                            "to": int(holder),
                            "team": int(current_team),
                        })
                    else:
                        interceptions.append({
                            "frame": int(frame_index),
                            "from_team": int(previous_team),
                            "to_team": int(current_team),
                            "from_player": int(previous_holder),
                            "to_player": int(holder),
                        })
                    last_event_frame[event_key] = frame_index

            previous_holder = holder
            previous_team = current_team
            previous_seen_frame = frame_index

        return passes, interceptions

    def detect_passes(self, possession_list, team_assignment_per_frame):
        return self.detect_passes_and_interceptions(
            possession_list, team_assignment_per_frame
        )[0]

    def detect_interception(self, possession_list, team_assignment_per_frame):
        return self.detect_passes_and_interceptions(
            possession_list, team_assignment_per_frame
        )[1]
