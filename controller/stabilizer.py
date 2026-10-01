"""Turns the noisy per-frame label into a stable gesture.

A new gesture is accepted only after it has been seen for N consecutive frames
(N per gesture, see config.HOLD_FRAMES). The current gesture is kept until a
replacement earns its frames, which prevents flicker (hysteresis).
"""
from ML_Module.config import DEFAULT_HOLD_FRAMES, IDLE


class GestureStabilizer:
    def __init__(self, hold_frames=None, default_hold=DEFAULT_HOLD_FRAMES, initial=IDLE):
        self.hold_frames = hold_frames or {}
        self.default_hold = default_hold
        self.initial = initial
        self.reset()

    def reset(self):
        self.current = self.initial
        self._candidate = None
        self._count = 0

    def update(self, label):
        if label == self.current:
            self._candidate, self._count = None, 0
            return self.current

        if label == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = label, 1

        if self._count >= self.hold_frames.get(label, self.default_hold):
            self.current = label
            self._candidate, self._count = None, 0
        return self.current
