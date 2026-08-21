import cv2

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from Camera.camera import open_camera, get_frame
from hand_tracking.hand_tracker import HandTracker
from Landmarks.landmarks_processor import process_landmarks

from features.feature_extractor import (
    extract_features,
    create_feature_vector
)

from features.feature_output import create_feature_output
from features.cv_pipeline import get_hand_features

def main():

    # Open camera
    cap = open_camera()

    if cap is None:
        print("Camera not opened.")
        return

    # Create hand tracker
    tracker = HandTracker()

    # Create display window
    cv2.namedWindow("FlowSync", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FlowSync", 1280, 720)

    while cap.isOpened():

        # Get frame from camera
        frame = get_frame(cap)

        if frame is None:
            continue

        # Mirror the camera
        frame = cv2.flip(frame, 1)

        # Detect hands
        result = tracker.detect(frame)

        # Process detected landmarks
        processed_landmarks = process_landmarks(result)

        # Process each detected hand
        for hand in processed_landmarks:

            output = get_hand_features(
                hand["landmarks"],
                hand["finger_status"]
            )   
            if output is not None:
                continue
                # Here you can use the output for further processing
                # For example, you can send it to an AI/ML model

            # CV output is now ready for the next layer
            # AI/ML layer will use this output later.

        # Draw hand landmarks
        if result and getattr(result, "hand_landmarks", None):

            for hand_landmarks in result.hand_landmarks:

                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

        # Display frame
        cv2.imshow("FlowSync", frame)

        # Space or ESC → exit
        if cv2.waitKey(1) & 0xFF in (ord(" "), 27):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()