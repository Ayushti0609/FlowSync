"""Record labelled gesture samples.

Every run of this script is ONE SESSION and is appended to dataset.csv with a
session id. Run it 3+ times (different day / lighting / distance) so the trainer
can test on whole sessions it has never seen.

    python ML_Module/collect_data.py                  # all gestures
    python ML_Module/collect_data.py --gestures idle  # only add some gestures
"""
import argparse
import os
import sys
import time
import shutil
from datetime import datetime

import cv2
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Camera.camera import open_camera, get_frame
from hand_tracking.hand_tracker import HandTracker
from Landmarks.landmarks_processor import process_landmarks
from features.feature_schema import FEATURE_COUNT

from ML_Module.config import DATASET_PATH, GESTURES, GESTURE_HINTS
from ML_Module.pipeline import FeaturePipeline
from ML_Module.viz import draw_hand_landmarks, draw_hud

COLUMNS = [f"f_{i}" for i in range(FEATURE_COUNT)] + ["label", "session"]
QUIT_KEYS = (27, ord("q"), ord("Q"))


def next_session_id(fresh=False):
    if fresh or not os.path.exists(DATASET_PATH):
        return 1
    df = pd.read_csv(DATASET_PATH)
    return 1 if "session" not in df.columns else int(df["session"].max()) + 1


def save_rows(rows, fresh=False):
    """Save rows. In fresh mode, preserve the previous CSV as a timestamped backup."""
    new = pd.DataFrame(rows, columns=COLUMNS)
    if fresh and os.path.exists(DATASET_PATH):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(os.path.dirname(DATASET_PATH), f"dataset_backup_{stamp}.csv")
        shutil.copy2(DATASET_PATH, backup)
        print(f"[BACKUP] Existing dataset preserved at: {backup}")
    if not fresh and os.path.exists(DATASET_PATH):
        old = pd.read_csv(DATASET_PATH)
        if "session" not in old.columns:  # file from the old collector
            old["session"] = 0
        if list(old.columns) != COLUMNS:
            raise SystemExit("Existing dataset.csv has different columns; move it aside first.")
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(DATASET_PATH, index=False)
    return new


def read_frame(cap, tracker):
    frame = get_frame(cap)
    if frame is None:
        return None, None
    frame = cv2.flip(frame, 1)  # same mirroring as the live loop
    return frame, tracker.detect(frame)


def wait_for_space(cap, tracker, gesture, idx, total):
    while True:
        frame, result = read_frame(cap, tracker)
        if frame is None:
            continue
        draw_hand_landmarks(frame, result)
        draw_hud(frame, [
            f"[{idx}/{total}] NEXT: {gesture.upper()}  (SPACE = start)",
            GESTURE_HINTS.get(gesture, ""),
        ], (0, 255, 255))
        cv2.imshow("FlowSync Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return True
        if key in QUIT_KEYS:
            return False


def countdown(cap, tracker, gesture, seconds=3):
    for sec in range(seconds, 0, -1):
        end = time.time() + 1.0
        while time.time() < end:
            frame, result = read_frame(cap, tracker)
            if frame is None:
                continue
            draw_hand_landmarks(frame, result)
            draw_hud(frame, [f"{gesture.upper()} starts in {sec}...", GESTURE_HINTS.get(gesture, "")],
                     (0, 0, 255))
            cv2.imshow("FlowSync Data Collector", frame)
            if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                return False
    return True


def record(cap, tracker, pipeline, gesture, session, frames):
    """Returns the rows for this gesture, or None if the user aborted."""
    rows = []
    pipeline.reset()  # no smoothing carry-over from the previous gesture
    while len(rows) < frames:
        frame, result = read_frame(cap, tracker)
        if frame is None:
            continue
        draw_hand_landmarks(frame, result)

        smoothed, output = pipeline.process(process_landmarks(result))
        color = (0, 255, 0)
        if smoothed is None:
            status, color = "No hand detected!", (0, 0, 255)
        elif output is None:
            status, color = "Hand features invalid - adjust hand", (0, 165, 255)
        else:
            rows.append(list(output["feature_vector"]) + [gesture, session])
            status = f"Recording {gesture.upper()}  {len(rows)}/{frames}"

        draw_hud(frame, [status, "Keep the pose; move/rotate slightly. ESC = abort"], color)
        cv2.imshow("FlowSync Data Collector", frame)
        if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
            return None
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gestures", nargs="+", help=f"subset of: {', '.join(GESTURES)}")
    parser.add_argument("--frames", type=int, default=300, help="valid frames per gesture")
    parser.add_argument("--fresh", action="store_true",
                        help="start a clean dataset; the previous CSV is backed up, not deleted")
    args = parser.parse_args()

    gestures = args.gestures or GESTURES
    if args.frames < 1:
        raise SystemExit("--frames must be at least 1")
    if args.fresh and set(gestures) != set(GESTURES):
        raise SystemExit("--fresh requires the complete gesture set. Omit --gestures when starting fresh.")
    unknown = [g for g in gestures if g not in GESTURES]
    if unknown:
        raise SystemExit(f"Unknown gestures: {unknown}. Valid: {GESTURES}")

    cap = open_camera()
    if cap is None:
        raise SystemExit("Camera could not be opened.")

    session = next_session_id(fresh=args.fresh)
    tracker = HandTracker()
    pipeline = FeaturePipeline()
    rows, aborted = [], False

    print(f"\nFLOWSYNC DATA COLLECTION - session {session}")
    print(f"{len(gestures)} gestures x {args.frames} frames. SPACE = start, ESC/Q = abort.\n")

    try:
        for i, gesture in enumerate(gestures, 1):
            if not wait_for_space(cap, tracker, gesture, i, len(gestures)):
                aborted = True
                break
            if not countdown(cap, tracker, gesture):
                aborted = True
                break
            gesture_rows = record(cap, tracker, pipeline, gesture, session, args.frames)
            if gesture_rows is None:
                aborted = True
                print(f"Aborted during {gesture}; its partial frames were discarded.")
                break
            rows.extend(gesture_rows)
            print(f"  saved {len(gesture_rows)} frames for {gesture}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if aborted and args.fresh:
        print("Fresh collection was aborted. Existing dataset was left untouched; restart with --fresh to try again.")
        return

    if rows:
        df = save_rows(rows, fresh=args.fresh)
        print(f"\n{'Partial s' if aborted else 'S'}ession {session} saved: {len(rows)} new rows "
              f"({len(df)} total) -> {DATASET_PATH}")
        if aborted:
            print("[NOTE] This was a partial session. For a clean fresh dataset, finish all gestures in the first run.")
    else:
        print("No samples recorded.")


if __name__ == "__main__":
    main()
