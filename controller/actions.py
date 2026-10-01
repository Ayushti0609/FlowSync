"""The only place that touches the operating system.

SystemActions  -> real mouse / keyboard (pyautogui)
DryRunActions  -> prints what WOULD happen; safe for testing the pipeline
"""

import os
from datetime import datetime

from ML_Module.config import SCREENSHOT_DIR


class SystemActions:
    def __init__(self):
        import pyautogui  # imported lazily so tests / dry-runs need no display

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
        self._pg = pyautogui

    def screen_size(self):
        size = self._pg.size()
        return int(size[0]), int(size[1])

    def position(self):
        p = self._pg.position()
        return int(p[0]), int(p[1])

    def move_to(self, x, y):
        self._pg.moveTo(x, y)

    def click(self):
        self._pg.click()

    def right_click(self):
        self._pg.rightClick()

    def mouse_down(self):
        self._pg.mouseDown()

    def mouse_up(self):
        self._pg.mouseUp()

    def scroll(self, amount):
        self._pg.scroll(int(amount))

    def volume_up(self):
        self._pg.press("volumeup")

    def volume_down(self):
        self._pg.press("volumedown")

    def screenshot(self):
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        path = os.path.join(SCREENSHOT_DIR, filename)
        self._pg.screenshot().save(path)
        return path


class DryRunActions:
    def __init__(self, screen=(1920, 1080), verbose=True):
        self._screen = screen
        self._pos = (screen[0] // 2, screen[1] // 2)
        self.verbose = verbose
        self.log = []

    def _do(self, name):
        self.log.append(name)
        if self.verbose and name != "move":
            print(f"[DRY-RUN] {name}")

    def screen_size(self):
        return self._screen

    def position(self):
        return self._pos

    def move_to(self, x, y):
        self._pos = (x, y)
        self._do("move")

    def click(self):
        self._do("click")

    def right_click(self):
        self._do("right_click")

    def mouse_down(self):
        self._do("mouse_down")

    def mouse_up(self):
        self._do("mouse_up")

    def scroll(self, amount):
        direction = "up" if amount > 0 else "down"
        self._do(f"scroll_{direction}_{abs(int(amount))}")

    def volume_up(self):
        self._do("volume_up")

    def volume_down(self):
        self._do("volume_down")

    def screenshot(self):
        self._do("screenshot")
        return "DRY RUN"
