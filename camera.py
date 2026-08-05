import os
import time
import urllib.request
import zipfile
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_ASSET_PATH = os.path.join(os.path.dirname(__file__) or ".", "hand_landmarker.task")

# Thread-safe global variables to share detection state with the main window thread
latest_result = None


def is_valid_task_file(path: str) -> bool:
    return zipfile.is_zipfile(path)


def ensure_model_asset(model_path: str = MODEL_ASSET_PATH) -> str:
    if os.path.exists(model_path) and is_valid_task_file(model_path):
        return model_path

    if os.path.exists(model_path):
        os.remove(model_path)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to download the hand landmark model: {exc}") from exc

    if not is_valid_task_file(model_path):
        raise RuntimeError("Downloaded hand_landmarker.task is not a valid task archive.")

    return model_path


# Callback function to receive detection results asynchronously from MediaPipe
def process_result(result: Any, output_image: Any, timestamp_ms: int):
    global latest_result
    latest_result = result


def open_camera():
    camera_candidates = [
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_DSHOW),
        (1, cv2.CAP_ANY),
        (2, cv2.CAP_DSHOW),
        (2, cv2.CAP_ANY),
    ]

    for index, backend in camera_candidates:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(8):
            success, frame = cap.read()
            if success and frame is not None and frame.size > 0:
                if np.mean(frame) > 5:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    return cap

        cap.release()

    return None


def main() -> None:
    model_asset_path = ensure_model_asset()

    # 1. Setup MediaPipe Options for Live Stream Mode
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=process_result,
        num_hands=2,
    )

    # 2. Initialize the detector and video capture
    with vision.HandLandmarker.create_from_options(options) as detector:
        cap = open_camera()

        if cap is None:
            print("Camera not opened. Try another camera index or connect a webcam.")
            return

        time.sleep(1)

        cv2.namedWindow("capture image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("capture image", 1280, 720)

        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None or frame.size == 0:
                print("frame not received.")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)  # horizontal camera flip

            # Convert frame from BGR to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Dispatch the image frame to the background thread tracker
            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(mp_image, timestamp_ms)

            # Draw landmarks onto the frame if any are currently detected
            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    # Draw connections (lines)
                    drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        vision.HandLandmarksConnections.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

            # Show the annotated window frame
            cv2.imshow("capture image", frame)

            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()