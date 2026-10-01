"""Single source of truth for the ML module and the controller.

Gesture names, thresholds and paths live here so the data collector, the
trainer, the live loop and the controller can never drift apart.
"""
import os

ML_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ML_DIR)

DATASET_PATH = os.path.join(ML_DIR, "dataset.csv")
MODEL_PATH = os.path.join(ML_DIR, "gesture_model.pkl")
SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "screenshots")
METRICS_PATH = os.path.join(ML_DIR, "metrics.json")

# ---------------------------------------------------------------- gestures
PALM = "palm"
POINTER = "pointer"
LEFT_CLICK = "left_click"
PEACE = "peace"          # V sign -> right click
SCREENSHOT = "screenshot"  # index+middle+ring up -> screenshot
FIST = "fist"
THUMB_UP = "thumb_up"
THUMB_DOWN = "thumb_down"
IDLE = "idle"

GESTURES = [PALM, POINTER, LEFT_CLICK, PEACE, SCREENSHOT, FIST, THUMB_UP, THUMB_DOWN, IDLE]
# Gestures that trigger a click when the pose starts: LEFT_CLICK = left click, PEACE = right click.
CLICK_GESTURES = (LEFT_CLICK, PEACE)

# Shown on screen while recording (short lines so they fit). Clean, HELD poses give a
# much better model.
GESTURE_HINTS = {
    PALM: ("Open hand, all 5 fingers spread.", "Cursor freezes."),
    POINTER: ("Only the INDEX finger out.", "Move your hand around."),
    LEFT_CLICK: ("LEFT CLICK = thumb + INDEX tips pinched.", "HOLD the pinch (do not tap)."),
    PEACE: ("RIGHT CLICK = peace / V sign.",
            "Index + middle UP and spread apart.",
            "Ring, pinky, thumb curled in. HOLD still."),
    SCREENSHOT: ("SCREENSHOT = 3 fingers up.",
                 "Index + middle + ring UP and spread.",
                 "Pinky and thumb curled in. HOLD still."),
    FIST: ("Tight closed fist = SCROLL.", "Move your fist up/down slowly to scroll; hold still to stop."),
    THUMB_UP: ("Fist with thumb pointing UP.",),
    THUMB_DOWN: ("Fist with thumb pointing DOWN.",),
    IDLE: ("Relaxed hand, random in-between poses,",
           "transitions. Vary it. No fists, no gestures!"),
}

# ------------------------------------------------------- shared processing
# One smoothing value for BOTH data collection and live use (they used to differ).
SMOOTHING_ALPHA = 0.60

# Frames below this model confidence are treated as IDLE.
CONFIDENCE_THRESHOLD = 0.55

# How many consecutive frames a NEW gesture must be seen before it is accepted.
DEFAULT_HOLD_FRAMES = 3
HOLD_FRAMES = {
    POINTER: 2,
    IDLE: 2,
    SCREENSHOT: 4,   # 3-finger pose is close to peace/palm: demand a steadier hold
}
