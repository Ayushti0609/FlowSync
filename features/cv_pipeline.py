from features.feature_extractor import (
    extract_features,
    create_feature_vector
)

from features.feature_output import create_feature_output


def get_hand_features(landmarks, finger_status):
    """
    Converts processed hand landmarks into the final
    CV output required by the AI/ML layer.
    """

    features = extract_features(
        landmarks,
        finger_status
    )

    if features is None:
        return None

    feature_vector = create_feature_vector(features)

    output = create_feature_output(
        feature_vector,
        features["finger_states"]
    )

    return output