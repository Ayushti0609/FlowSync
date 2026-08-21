# This file processes and validates hand landmarks detected by MediaPipe.

LANDMARK_NAMES = {
    0: "Wrist",

    1: "Thumb_CMC",
    2: "Thumb_MCP",
    3: "Thumb_IP",
    4: "Thumb_Tip",

    5: "Index_MCP",
    6: "Index_PIP",
    7: "Index_DIP",
    8: "Index_Tip",

    9: "Middle_MCP",
    10: "Middle_PIP",
    11: "Middle_DIP",
    12: "Middle_Tip",

    13: "Ring_MCP",
    14: "Ring_PIP",
    15: "Ring_DIP",
    16: "Ring_Tip",

    17: "Pinky_MCP",
    18: "Pinky_PIP",
    19: "Pinky_DIP",
    20: "Pinky_Tip"
}


FINGER_LANDMARKS = {
    "Thumb": [
        "Thumb_CMC",
        "Thumb_MCP",
        "Thumb_IP",
        "Thumb_Tip"
    ],

    "Index": [
        "Index_MCP",
        "Index_PIP",
        "Index_DIP",
        "Index_Tip"
    ],

    "Middle": [
        "Middle_MCP",
        "Middle_PIP",
        "Middle_DIP",
        "Middle_Tip"
    ],

    "Ring": [
        "Ring_MCP",
        "Ring_PIP",
        "Ring_DIP",
        "Ring_Tip"
    ],

    "Pinky": [
        "Pinky_MCP",
        "Pinky_PIP",
        "Pinky_DIP",
        "Pinky_Tip"
    ]
}


def is_valid_landmark(landmark):
    """
    Checks whether a MediaPipe landmark contains
    valid normalized x and y coordinates.
    """

    if landmark is None:
        return False

    if not (0.0 <= landmark.x <= 1.0):
        return False

    if not (0.0 <= landmark.y <= 1.0):
        return False

    return True


def check_finger_visibility(landmarks):
    """
    Determines whether each finger has all of its
    required landmarks available.
    """

    finger_status = {}

    for finger, required_landmarks in FINGER_LANDMARKS.items():

        available_count = 0

        for landmark_name in required_landmarks:
            if landmarks.get(landmark_name) is not None:
                available_count += 1

        total_landmarks = len(required_landmarks)

        finger_status[finger] = {
            "available": available_count > 0,
            "complete": available_count == total_landmarks,
            "landmarks_found": available_count,
            "landmarks_required": total_landmarks
        }

    return finger_status


def process_landmarks(result):

    processed_hands = []

    if not result or not result.hand_landmarks:
        return processed_hands

    for hand_landmarks in result.hand_landmarks:

        landmarks = {}

        for index, landmark in enumerate(hand_landmarks):

            name = LANDMARK_NAMES[index]

            if not is_valid_landmark(landmark):
                landmarks[name] = None
                continue

            landmarks[name] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            }

        finger_status = check_finger_visibility(landmarks)

        processed_hands.append({
            "landmarks": landmarks,
            "finger_status": finger_status
        })

    return processed_hands