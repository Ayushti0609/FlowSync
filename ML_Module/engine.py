"""Frame in -> gesture observation out. Contains no OS actions."""
from dataclasses import dataclass

from features.feature_schema import LANDMARK_FEATURES

from ML_Module.classifier import GestureClassifier
from ML_Module.config import (
    CONFIDENCE_THRESHOLD, IDLE, MODEL_PATH, SCREENSHOT, THUMB_DOWN, THUMB_UP,
)
from ML_Module.pipeline import FeaturePipeline
from ML_Module.screenshot_detector import is_screenshot_pose

# Normalised thumb-tip Y (wrist is the origin; image Y grows downward).
# Thumb up => negative, thumb down => positive. Verified on the recorded data.
_THUMB_TIP_Y = LANDMARK_FEATURES.index("Thumb_Tip") * 3 + 1


@dataclass
class Observation:
    label: str         # final label after confidence + sanity checks
    confidence: float
    raw_label: str     # what the model said before those checks
    landmarks: dict    # smoothed landmarks (CV-module naming)
    valid: bool        # False when features could not be computed


class GestureEngine:
    def __init__(self, model_path=MODEL_PATH):
        self.classifier = GestureClassifier(model_path)
        self.pipeline = FeaturePipeline()

    def process(self, hands):
        """hands: output of process_landmarks(). Returns Observation or None (no hand)."""
        smoothed, output = self.pipeline.process(hands)

        if smoothed is None:
            return None
        if output is None:
            return Observation(IDLE, 0.0, "invalid", smoothed, False)

        fv = output["feature_vector"]
        raw, conf = self.classifier.predict(fv)

        if is_screenshot_pose(smoothed):
            raw, conf, label = SCREENSHOT, 1.0, SCREENSHOT
        else:
            label = raw if conf >= CONFIDENCE_THRESHOLD else IDLE
            if label == SCREENSHOT:
                label = IDLE
            label = self._thumb_sanity(label, fv)
        return Observation(label, conf, raw, smoothed, True)

    @staticmethod
    def _thumb_sanity(label, fv):
        """Stops random poses from spamming the volume keys."""
        ty = fv[_THUMB_TIP_Y]
        if label == THUMB_UP and ty >= 0:
            return IDLE
        if label == THUMB_DOWN and ty <= 0:
            return IDLE
        return label
