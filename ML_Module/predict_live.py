"""Live loop: camera -> engine -> stabilizer -> controller.

    python ML_Module/predict_live.py --dry-run   # watch gestures, no mouse control
    python ML_Module/predict_live.py             # real control

Keys (in the window):  P = pause/resume control    SPACE / ESC = quit
"""
import argparse
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Camera.camera import open_camera, get_frame
from hand_tracking.hand_tracker import HandTracker
from Landmarks.landmarks_processor import process_landmarks

from controller import DryRunActions, GestureController, GestureStabilizer, SystemActions
from ML_Module.config import HOLD_FRAMES, IDLE, MODEL_PATH
from ML_Module.engine import GestureEngine
from ML_Module.viz import draw_hand_landmarks, draw_hud


def run(dry_run=False, show_landmarks=True):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"{MODEL_PATH} not found. Run train_model.py first.")

    engine = GestureEngine()
    stabilizer = GestureStabilizer(HOLD_FRAMES)
    actions = DryRunActions() if dry_run else SystemActions()
    controller = GestureController(actions)

    cap = open_camera()
    if cap is None:
        raise SystemExit("Camera could not be opened.")
    tracker = HandTracker()

    print("FlowSync running" + (" (DRY-RUN)" if dry_run else "") + ".  P = pause, SPACE/ESC = quit")
    prev_t = time.time()
    fps = 0.0

    try:
        while cap.isOpened():
            frame = get_frame(cap)
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)

            result = tracker.detect(frame)
            obs = engine.process(process_landmarks(result))

            stable = stabilizer.update(obs.label if obs else IDLE)
            controller.update(stable, obs.landmarks if obs else None)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
            prev_t = now

            if show_landmarks:
                draw_hand_landmarks(frame, result)

            title = f"GESTURE: {stable.upper()}"
            if controller.is_scrolling:
                title += "  [SCROLL]"
            if not controller.enabled:
                title = "PAUSED  (press P)"
            raw = f"model: {obs.raw_label} {obs.confidence:.0%}" if obs else "no hand"
            text, t = controller.last_action
            extra = text if now - t < 1.0 else ""
            color = (0, 0, 255) if controller.is_scrolling or not controller.enabled else (0, 255, 0)
            draw_hud(frame, [title, f"{raw}   {extra}   {fps:.0f} fps"], color)

            cv2.imshow("FlowSync Controller", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord(" ")):
                break
            if key in (ord("p"), ord("P")):
                print("[CONTROL]", "resumed" if controller.toggle() else "paused")
    finally:
        controller.release_all()  # never leave the mouse button held down
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print actions instead of performing them")
    parser.add_argument("--no-landmarks", action="store_true", help="do not draw hand skeleton")
    args = parser.parse_args()
    run(dry_run=args.dry_run, show_landmarks=not args.no_landmarks)
