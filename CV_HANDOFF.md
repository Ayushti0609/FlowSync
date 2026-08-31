# FlowSync Computer Vision Module

## Purpose

The Computer Vision module captures hand data from the camera,
detects hand landmarks using MediaPipe, processes the landmarks,
and converts them into a fixed numerical feature representation.

The CV module does NOT classify gestures.

Gesture classification and machine learning are handled by the AI/ML layer.

---

## CV Pipeline

Camera
↓
Hand Detection
↓
21 Hand Landmarks
↓
Landmark Processing
↓
Normalization
↓
Distance Features
↓
Angle Features
↓
Finger States
↓
79-Feature Vector
↓
AI/ML Layer

---

## Public Interface

The AI/ML layer should use:

```python
from features.cv_pipeline import get_hand_features

output = get_hand_features(
    landmarks,
    finger_status
)