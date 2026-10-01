"""Loads the trained model and turns a feature vector into (label, confidence)."""
import joblib
import numpy as np

from features.feature_schema import FEATURE_COUNT
from ML_Module.config import MODEL_PATH, SMOOTHING_ALPHA


class GestureClassifier:
    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)

        if isinstance(bundle, dict):
            self.model = bundle["model"]
            self.meta = {k: v for k, v in bundle.items() if k != "model"}
        else:  # plain estimator saved by the old trainer
            self.model = bundle
            self.meta = {}

        n_in = getattr(self.model, "n_features_in_", FEATURE_COUNT)
        if n_in != FEATURE_COUNT:
            raise ValueError(
                f"Model expects {n_in} features but the CV module produces "
                f"{FEATURE_COUNT}. Re-train with train_model.py."
            )

        saved_alpha = self.meta.get("smoothing_alpha")
        if saved_alpha is not None and abs(saved_alpha - SMOOTHING_ALPHA) > 1e-9:
            print(
                f"[WARN] Model was trained with smoothing alpha={saved_alpha} but "
                f"config uses {SMOOTHING_ALPHA}. Re-collect/re-train for best accuracy."
            )

        # Single-sample prediction is faster without a worker pool.
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1

        self.classes = [str(c) for c in self.model.classes_]

    def predict(self, feature_vector):
        x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        i = int(np.argmax(probs))
        return self.classes[i], float(probs[i])
