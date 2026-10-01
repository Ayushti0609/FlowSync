"""Train the gesture model and report honest, session-based accuracy.

    python ML_Module/train_model.py
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from features.feature_schema import FEATURE_COUNT
from ML_Module.config import (
    DATASET_PATH, GESTURES, IDLE, METRICS_PATH, MODEL_PATH, SMOOTHING_ALPHA,
)


def make_model():
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise SystemExit(f"{DATASET_PATH} not found. Run collect_data.py first.")
    df = pd.read_csv(DATASET_PATH)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    if len(feature_cols) != FEATURE_COUNT:
        raise SystemExit(
            f"Expected {FEATURE_COUNT} feature columns, found {len(feature_cols)}. "
            "Use the matching FlowSync collector and do not train on a different CSV format."
        )
    if "label" not in df.columns:
        raise SystemExit("Dataset must contain a 'label' column.")
    X = np.ascontiguousarray(df[feature_cols].to_numpy(dtype=np.float32))
    y = df["label"].astype(str).to_numpy()
    groups = df["session"].to_numpy() if "session" in df.columns else np.zeros(len(df), dtype=int)

    # Rows whose label is no longer a gesture (e.g. the old pinch-style "right_click")
    # are ignored, so stale recordings can't poison the model.
    keep = np.isin(y, GESTURES)
    if not keep.all():
        stale = pd.Series(y[~keep]).value_counts().to_dict()
        print(f"[INFO] Ignoring {int((~keep).sum())} rows with labels not in config.GESTURES: {stale}\n")
        X, y, groups = X[keep], y[keep], groups[keep]
    return X, y, groups


def evaluate(X, y, groups):
    """Test on data the model has not seen: whole sessions if we have several,
    otherwise the last 20% of each class (webcam frames are near-duplicates, so a
    random split would report inflated accuracy)."""
    sessions = np.unique(groups)
    if len(sessions) >= 2:
        pred = cross_val_predict(make_model(), X, y, groups=groups, cv=LeaveOneGroupOut())
        return y, pred, f"leave-one-session-out ({len(sessions)} sessions)", groups

    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        cut = int(len(idx) * 0.8)
        train_idx += idx[:cut].tolist()
        test_idx += idx[cut:].tolist()
    model = make_model().fit(X[train_idx], y[train_idx])
    return y[test_idx], model.predict(X[test_idx]), "time-ordered holdout (only 1 session - record more!)", None


def per_session_table(y_true, y_pred, eval_groups, labels):
    """Recall of every class when each session is the one held out.
    A low row/column pinpoints the session or gesture that was recorded differently."""
    table = {}
    for sess in np.unique(eval_groups):
        m = eval_groups == sess
        row = {}
        for label in labels:
            lm = m & (y_true == label)
            row[label] = float((y_pred[lm] == label).mean()) if lm.any() else np.nan
        row["ALL"] = float((y_pred[m] == y_true[m]).mean())
        table[f"session {sess}"] = row
    return pd.DataFrame(table).T.round(2)


def main():
    X, y, groups = load_dataset()
    print(f"Samples: {len(X)}   Features: {X.shape[1]}   Sessions: {len(np.unique(groups))}")
    print(pd.Series(y).value_counts().to_string(), "\n")

    missing = [g for g in GESTURES if g not in set(y)]
    if missing:
        print(f"[WARN] no samples yet for: {missing}\n")
    if IDLE not in y:
        raise SystemExit("Dataset is missing 'idle' samples. Record every gesture, including relaxed/transitional poses.")
    if missing:
        raise SystemExit(f"Dataset is incomplete. Missing gesture classes: {missing}. Record the full gesture set before training.")
    if len(np.unique(groups)) < 2:
        print("[WARN] Only one recording session found. Accuracy may be optimistic; record a second full session with another hand/person or in a different setup.\n")

    y_true, y_pred, mode, eval_groups = evaluate(X, y, groups)
    acc = accuracy_score(y_true, y_pred)
    labels = sorted(set(y))
    print(f"Evaluation: {mode}\nAccuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string(), "\n")

    if eval_groups is not None:
        print("Per-session recall (that session was held out):")
        print(per_session_table(y_true, y_pred, eval_groups, labels).to_string(), "\n")

    model = make_model().fit(X, y)  # final model uses ALL data
    joblib.dump({
        "model": model,
        "classes": [str(c) for c in model.classes_],
        "feature_count": int(X.shape[1]),
        "smoothing_alpha": SMOOTHING_ALPHA,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "evaluation": mode,
            "accuracy": acc,
            "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
            "labels": labels,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)

    print(f"Model saved   -> {MODEL_PATH}\nMetrics saved -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
