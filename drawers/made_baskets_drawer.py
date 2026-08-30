import cv2


class MadeBasketsDrawer:
    """Draw a visible made-basket banner and cumulative count."""

    def __init__(self, color=(0, 200, 0), radius=18, text_color=(255, 255, 255), display_frames=45):
        self.color = color
        self.radius = radius
        self.text_color = text_color
        self.display_frames = display_frames

    def draw(self, frames, made_baskets):
        events = sorted(made_baskets, key=lambda event: int(event.get("frame", -1)))
        for frame_index, frame in enumerate(frames):
            completed = [event for event in events if int(event.get("frame", -1)) <= frame_index]
            active = [
                event for event in completed
                if frame_index - int(event.get("frame", -1)) < self.display_frames
            ]

            if completed:
                cv2.putText(
                    frame,
                    f"Made baskets: {len(completed)}",
                    (frame.shape[1] // 2 - 100, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    self.text_color,
                    2,
                    cv2.LINE_AA,
                )

            if not active:
                continue

            event = active[-1]
            center = event.get("center")
            if center is not None:
                x, y = int(center[0]), int(center[1])
                cv2.circle(frame, (x, y), self.radius, self.color, cv2.FILLED)
                cv2.circle(frame, (x, y), self.radius, (0, 0, 0), 2)

            label = "MADE BASKET"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            x1 = frame.shape[1] // 2 - text_size[0] // 2 - 15
            y1 = 48
            x2 = x1 + text_size[0] + 30
            y2 = 88
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color, cv2.FILLED)
            cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
            cv2.putText(
                frame,
                label,
                (x1 + 15, y2 - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                self.text_color,
                2,
                cv2.LINE_AA,
            )
        return frames
