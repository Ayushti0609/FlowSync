"""Landmarks -> smoothed landmarks -> feature vector.

Used by BOTH the data collector and the live loop, so the model always sees
features that were produced exactly the way the training data was.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Landmarks.landmarks_processor import check_finger_visibility
from Landmarks.landmarks_smoothing import LandmarkSmoother
from features.cv_pipeline import get_hand_features

from ML_Module.config import SMOOTHING_ALPHA


class FeaturePipeline:
    def __init__(self, alpha=SMOOTHING_ALPHA):
        self.smoother = LandmarkSmoother(alpha=alpha)

    def reset(self):
        self.smoother.reset()

    def process(self, hands):
        """hands: output of process_landmarks().

        Returns (smoothed_landmarks, feature_output):
          (None, None)       -> no hand in frame
          (landmarks, None)  -> hand found but features invalid (e.g. out of frame)
          (landmarks, output)-> ok; output["feature_vector"] has FEATURE_COUNT values
        Only the first detected hand is used.
        """
        if not hands:
            self.smoother.reset()  # don't blend a new hand with a stale one
            return None, None

        smoothed = self.smoother.smooth(hands[0]["landmarks"])
        status = check_finger_visibility(smoothed)
        output = get_hand_features(smoothed, status)

        if output is None or not output.get("valid"):
            return smoothed, None
        return smoothed, output
