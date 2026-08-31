# Defines the fixed order of features used by FlowSync.

LANDMARK_FEATURES = [
    "Wrist",
    
    "Thumb_CMC",
    "Thumb_MCP",
    "Thumb_IP",
    "Thumb_Tip",

    "Index_MCP",
    "Index_PIP",
    "Index_DIP",
    "Index_Tip",

    "Middle_MCP",
    "Middle_PIP",
    "Middle_DIP",
    "Middle_Tip",

    "Ring_MCP",
    "Ring_PIP",
    "Ring_DIP",
    "Ring_Tip",

    "Pinky_MCP",
    "Pinky_PIP",
    "Pinky_DIP",
    "Pinky_Tip"
]


DISTANCE_FEATURES = [
    "thumb_tip_to_index_tip",
    "index_tip_to_middle_tip",
    "middle_tip_to_ring_tip",
    "ring_tip_to_pinky_tip",

    "wrist_to_index_tip",
    "wrist_to_middle_tip",
    "wrist_to_ring_tip",
    "wrist_to_pinky_tip"
]


ANGLE_FEATURES = [
    "index_pip_angle",
    "index_dip_angle",

    "middle_pip_angle",
    "middle_dip_angle",

    "ring_pip_angle",
    "ring_dip_angle",

    "pinky_pip_angle",
    "pinky_dip_angle"
]


def get_feature_schema():

    schema = []

    index = 0

    # Landmark coordinates
    for landmark in LANDMARK_FEATURES:

        schema.append({
            "index": index,
            "name": f"{landmark}_X",
            "type": "coordinate"
        })
        index += 1

        schema.append({
            "index": index,
            "name": f"{landmark}_Y",
            "type": "coordinate"
        })
        index += 1

        schema.append({
            "index": index,
            "name": f"{landmark}_Z",
            "type": "coordinate"
        })
        index += 1

    # Distance features
    for distance in DISTANCE_FEATURES:

        schema.append({
            "index": index,
            "name": distance,
            "type": "distance"
        })
        index += 1

    # Angle features
    for angle in ANGLE_FEATURES:

        schema.append({
            "index": index,
            "name": angle,
            "type": "angle"
        })
        index += 1

    return schema


FEATURE_SCHEMA = get_feature_schema()

FEATURE_COUNT = len(FEATURE_SCHEMA)

#this function validates the feature vector to ensure it has the correct length and contains only numeric values
def validate_feature_vector(feature_vector):
    if feature_vector is None:
        return False

    if len(feature_vector) != FEATURE_COUNT:
        return False

    for value in feature_vector:
        if not isinstance(value, (int, float)):
            return False

    return True