from features.feature_schema import validate_feature_vector


def create_feature_output(feature_vector, finger_states):
    """
    Creates the final output of the CV feature pipeline.

    The output can be consumed by the AI/ML layer.
    """

    if feature_vector is None:
        return None

    if not validate_feature_vector(feature_vector):
        return None

    return {
        "feature_vector": feature_vector,
        "finger_states": finger_states,
        "valid": True
    }