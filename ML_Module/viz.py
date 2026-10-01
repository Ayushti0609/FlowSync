"""Small drawing helpers shared by the collector and the live loop."""
import cv2

try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
except ImportError:  # drawing is optional
    vision = None

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hand_landmarks(frame, result):
    if vision is None or not result or not getattr(result, "hand_landmarks", None):
        return
    for hand_landmarks in result.hand_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )


def draw_hud(frame, lines, color=(0, 255, 0)):
    height = 16 + 28 * len(lines)
    cv2.rectangle(frame, (10, 10), (620, height), (30, 30, 30), -1)
    for i, line in enumerate(lines):
        c = color if i == 0 else (200, 200, 200)
        scale = 0.8 if i == 0 else 0.55
        cv2.putText(
            frame,
            str(line),
            (20, 40 + 28 * i),
            FONT,
            scale,
            c,
            2 if i == 0 else 1
        )
