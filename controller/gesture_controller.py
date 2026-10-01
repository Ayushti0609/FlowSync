"""Maps a STABLE gesture to actions. No ML, no camera, no OS calls of its own.

  pointer      -> cursor follows the hand
  fist         -> move up/down to choose a continuous scroll direction; release to stop
  left_click   -> one left click when the pose starts
  peace (V)    -> one right click when the pose starts
  screenshot   -> one screenshot (3 fingers up) when the pose starts
  thumb_up     -> volume up, repeats while held
  thumb_down   -> volume down, repeats while held
  palm / idle  -> cursor frozen (clutch)
"""
from copy import error
import math
import time
import traceback
from collections import deque
from dataclasses import dataclass

from ML_Module.config import (
    CLICK_GESTURES, FIST, LEFT_CLICK, POINTER, SCREENSHOT, THUMB_DOWN, THUMB_UP,
)

MOVE_GESTURES = (POINTER,)  # fist is reserved for scrolling, not cursor movement
ONE_SHOT_GESTURES = (*CLICK_GESTURES, SCREENSHOT)  # fire once when the pose starts



@dataclass
class ControllerSettings:
    # Knuckle barely moves while pinching, so clicks land where you aimed.
    # Switch to "Index_Tip" if you prefer the cursor to sit on the fingertip.
    cursor_landmark: str = "Index_MCP"
    margin: float = 0.15            # ignore outer 15% of the camera view (reach corners)
    smoothing: float = 0.35         # 0..1, higher = snappier, lower = smoother
    cursor_sensitivity: float = 1.35  # how much the cursor moves relative to the hand
    deadzone_px: int = 2            # ignore sub-pixel jitter
    click_cooldown_s: float = 0.35
    click_rearm_s: float = 0.20     # pose must be released this long before it can fire again
    screenshot_cooldown_s: float = 1.5
    click_rewind_s: float = 0.20    # click where the cursor was just before the pinch began
    volume_interval_s: float = 0.18
    scroll_interval_s: float = 0.04 # time between continuous scroll ticks
    scroll_amount: int = 15         # scroll units per tick
    scroll_deadzone: float = 0.035  # normalized wrist movement needed to set/reverse direction
    hand_lost_grace_s: float = 0.40  # retained for compatibility; scrolling stops immediately on hand loss


class GestureController:
    def __init__(self, actions, settings=None):
        self.actions = actions
        self.s = settings or ControllerSettings()
        self.screen_w, self.screen_h = actions.screen_size()

        self.enabled = True
        self.is_dragging = False
        self.last_action = ("", 0.0)  # (text, timestamp) for the on-screen HUD

        self._latched = None          # click pose already handled (re-arms after a real release)
        self._released_since = None
        self._last_click = 0.0
        self._last_screenshot = -1e9
        self._last_volume = 0.0
        self._lost_since = None
        self._not_fist_since = None
        self._scroll_direction = None  # +1 = up, -1 = down
        self._scroll_anchor_y = None
        self._scroll_extreme_y = None
        self._last_scroll = 0.0
        self._fx = self._fy = None    # smoothed cursor position (float)
        self._sent = None             # last position actually sent to the OS
        self._history = deque(maxlen=90)  # (time, x, y) of recent cursor positions

    @property
    def is_scrolling(self):
        """Whether a continuous scroll direction is currently active."""
        return self._scroll_direction is not None

    # ------------------------------------------------------------ public API
    def update(self, gesture, landmarks, now=None):
        """gesture: stabilised label. landmarks: smoothed landmarks, or None if no hand."""
        now = time.time() if now is None else now

        if not self.enabled:
            self.release_all()
            return

        if landmarks is None:
            self._stop_scroll()  # no hand means no continued scrolling
            self._on_hand_lost(now)
            return
        self._lost_since = None

        if gesture in MOVE_GESTURES:
            pt = self._cursor_point(landmarks)
            if pt is not None:
                self._move(pt, now)

        self._handle_click(gesture, now)
        self._handle_scroll(gesture, landmarks, now)
        self._handle_volume(gesture, now)

    def release_all(self):
        """Always leave the mouse in a safe state."""
        if self.is_dragging:
            self.actions.mouse_up()
            self.is_dragging = False
        self._not_fist_since = None
        self._stop_scroll()
        self._latched = None
        self._released_since = None

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.release_all()
        return self.enabled

    # ----------------------------------------------------------------- cursor
    def _cursor_point(self, landmarks):
        p = landmarks.get(self.s.cursor_landmark)
        return (p["x"], p["y"]) if p else None

    # def _to_screen(self, x, y):
    #     m = self.s.margin
    #     span = 1.0 - 2 * m
    #     nx = min(1.0, max(0.0, (x - m) / span))
    #     ny = min(1.0, max(0.0, (y - m) / span))
    #     return 5 + nx * (self.screen_w - 10), 5 + ny * (self.screen_h - 10)
    def _to_screen(self, x, y):
        m = self.s.margin
        span = 1.0 - 2 * m

        nx = min(1.0, max(0.0, (x - m) / span))
        ny = min(1.0, max(0.0, (y - m) / span))

        # Apply cursor sensitivity around the screen center
        sensitivity = self.s.cursor_sensitivity
        nx = 0.5 + (nx - 0.5) * sensitivity
        ny = 0.5 + (ny - 0.5) * sensitivity

        # Keep cursor within screen boundaries
        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))

        return (
            5 + nx * (self.screen_w - 10),
            5 + ny * (self.screen_h - 10)
        )
        
    def _move(self, pt, now):
        tx, ty = self._to_screen(*pt)

        if self._fx is None:  # first move: start from the real cursor position
            self._fx, self._fy = self.actions.position()
            self._sent = (int(self._fx), int(self._fy))

        self._fx += (tx - self._fx) * self.s.smoothing
        self._fy += (ty - self._fy) * self.s.smoothing

        x, y = int(self._fx), int(self._fy)
        if math.hypot(x - self._sent[0], y - self._sent[1]) >= self.s.deadzone_px:
            self.actions.move_to(x, y)
            self._sent = (x, y)
        self._history.append((now, self._sent[0], self._sent[1]))

    def _rewind_position(self, now):
        cutoff = now - self.s.click_rewind_s
        for t, x, y in self._history:
            if t >= cutoff:
                return x, y
        return None

    # ------------------------------------------------------------------ clicks
    def _handle_click(self, gesture, now):
        if gesture in ONE_SHOT_GESTURES:
            self._released_since = None
            if gesture != self._latched:       # pose just started
                self._latched = gesture
                if gesture == SCREENSHOT:
                    if now - self._last_screenshot >= self.s.screenshot_cooldown_s:
                        self._fire_screenshot(now)
                elif now - self._last_click >= self.s.click_cooldown_s:
                    self._fire_click(gesture, now)
        elif self._latched is not None:
            # A 1-2 frame flicker out of the pose must not cause a second click:
            # only re-arm once the pose has really been released.
            if self._released_since is None:
                self._released_since = now
            if now - self._released_since >= self.s.click_rearm_s:
                self._latched = None
                self._released_since = None

    def _fire_click(self, gesture, now):
        pos = self._rewind_position(now)
        if pos is not None:
            self.actions.move_to(*pos)
            self._fx, self._fy = pos
            self._sent = pos
        self._history.clear()

        if gesture == LEFT_CLICK:
            self.actions.click()
            self.last_action = ("LEFT CLICK", now)
        else:
            self.actions.right_click()
            self.last_action = ("RIGHT CLICK", now)
        self._last_click = now

    def _fire_screenshot(self, now):
        self._last_screenshot = now
        try:
            path = self.actions.screenshot()
            if path == "DRY RUN":
                self.last_action = ("SCREENSHOT (DRY RUN)", now)
            else:
                self.last_action = ("SCREENSHOT CAPTURED", now)
                print(f"[ACTION] Screenshot saved: {path}")
        except Exception as error:  # never crash the live loop
            # self.last_action = ("SCREENSHOT FAILED", now)
            # print(f"[ACTION] Screenshot failed: {error}")
            self.last_action = ("SCREENSHOT FAILED", now)
            print(f"[ACTION] Screenshot failed: {type(error).__name__}: {error}")
            traceback.print_exc()

    # ---------------------------------------------------------- continuous scroll
    def _stop_scroll(self):
        self._scroll_direction = None
        self._scroll_anchor_y = None
        self._scroll_extreme_y = None

    def _handle_scroll(self, gesture, landmarks, now):
        """Fist movement selects direction; holding fist repeats scrolling.

        MediaPipe y coordinates increase downward. A negative y movement means
        the hand moved up, so PyAutoGUI receives a positive scroll amount.
        Direction stays latched while the hand is held still and reverses only
        after movement beyond the configured deadzone.
        """
        if gesture != FIST:
            self._stop_scroll()
            return

        wrist = landmarks.get("Wrist")
        if not wrist or wrist.get("y") is None:
            self._stop_scroll()
            return

        y = float(wrist["y"])
        if self._scroll_anchor_y is None:
            self._scroll_anchor_y = y
            self._scroll_extreme_y = y
            self.last_action = ("FIST: move hand up/down to scroll", now)
            return

        threshold = self.s.scroll_deadzone
        if self._scroll_direction is None:
            delta = y - self._scroll_anchor_y
            if delta <= -threshold:
                self._scroll_direction = 1  # hand moved up -> page scrolls up
                self._scroll_extreme_y = y
            elif delta >= threshold:
                self._scroll_direction = -1 # hand moved down -> page scrolls down
                self._scroll_extreme_y = y
        elif self._scroll_direction == 1:
            if y < self._scroll_extreme_y:
                self._scroll_extreme_y = y
            elif y - self._scroll_extreme_y >= threshold:
                self._scroll_direction = -1
                self._scroll_extreme_y = y
        else:  # currently scrolling down
            if y > self._scroll_extreme_y:
                self._scroll_extreme_y = y
            elif self._scroll_extreme_y - y >= threshold:
                self._scroll_direction = 1
                self._scroll_extreme_y = y

        if self._scroll_direction is None:
            return
        if now - self._last_scroll < self.s.scroll_interval_s:
            return

        amount = self.s.scroll_amount * self._scroll_direction
        try:
            self.actions.scroll(amount)
            direction_text = "UP" if self._scroll_direction == 1 else "DOWN"
            self.last_action = (f"SCROLL {direction_text}", now)
        except AttributeError:
            # Clear feedback for older/custom action adapters without scroll().
            self.last_action = ("SCROLL ACTION MISSING", now)
        self._last_scroll = now

    # ------------------------------------------------------------------ volume
    def _handle_volume(self, gesture, now):
        if gesture not in (THUMB_UP, THUMB_DOWN):
            return
        if now - self._last_volume < self.s.volume_interval_s:
            return
        if gesture == THUMB_UP:
            self.actions.volume_up()
            self.last_action = ("VOLUME UP", now)
        else:
            self.actions.volume_down()
            self.last_action = ("VOLUME DOWN", now)
        self._last_volume = now

    # --------------------------------------------------------------- hand lost
    def _on_hand_lost(self, now):
        if self._lost_since is None:
            self._lost_since = now
        self._stop_scroll()
        if now - self._lost_since >= self.s.hand_lost_grace_s:
            self.release_all()
