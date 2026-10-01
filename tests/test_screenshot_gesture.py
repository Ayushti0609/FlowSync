from PIL import Image

from controller.actions import DryRunActions
from controller.gesture_controller import GestureController
from features.feature_schema import FEATURE_COUNT
from ML_Module.config import FIST, IDLE, LEFT_CLICK, PEACE, POINTER, SCREENSHOT, THUMB_DOWN, THUMB_UP
from ML_Module.engine import GestureEngine
from ML_Module.screenshot_detector import is_screenshot_pose


def screenshot_pose():
    landmarks = {
        "Wrist": (0.0, 0.0, 0.0),
        "Thumb_CMC": (-0.2, 0.2, 0.0),
        "Thumb_MCP": (-0.45, 0.5, 0.0),
        "Thumb_IP": (-0.2, 0.65, 0.0),
        "Thumb_Tip": (0.0, 0.6, 0.0),
    }
    for finger, x in (("Index", -0.3), ("Middle", 0.0), ("Ring", 0.3)):
        landmarks.update({
            f"{finger}_MCP": (x, 1.0, 0.0),
            f"{finger}_PIP": (x, 2.0, 0.0),
            f"{finger}_DIP": (x, 3.0, 0.0),
            f"{finger}_Tip": (x, 4.0, 0.0),
        })
    landmarks.update({
        "Pinky_MCP": (0.6, 1.0, 0.0),
        "Pinky_PIP": (0.6, 1.4, 0.0),
        "Pinky_DIP": (1.0, 1.4, 0.0),
        "Pinky_Tip": (1.0, 1.0, 0.0),
    })
    return landmarks


def as_landmark_dict(points):
    return {name: dict(zip(("x", "y", "z"), point)) for name, point in points.items()}


class FakeActions:
    def __init__(self):
        self.calls = []

    def screen_size(self):
        return 1920, 1080

    def position(self):
        return 500, 500

    def move_to(self, *args):
        self.calls.append("move")

    def click(self):
        self.calls.append("click")

    def right_click(self):
        self.calls.append("right_click")

    def mouse_down(self):
        self.calls.append("mouse_down")

    def mouse_up(self):
        self.calls.append("mouse_up")

    def volume_up(self):
        self.calls.append("volume_up")

    def volume_down(self):
        self.calls.append("volume_down")

    def screenshot(self):
        self.calls.append("screenshot")
        return "screenshot.png"


def test_pose_uses_rotation_invariant_joint_geometry():
    pose = as_landmark_dict(screenshot_pose())
    assert is_screenshot_pose(pose)

    rotated = {
        name: {"x": -point["y"], "y": point["x"], "z": point["z"]}
        for name, point in pose.items()
    }
    assert is_screenshot_pose(rotated)
    assert not is_screenshot_pose({"Wrist": pose["Wrist"]})
    assert FEATURE_COUNT == 79


def test_engine_only_accepts_screenshot_model_label_with_matching_pose():
    class FakeClassifier:
        def predict(self, feature_vector):
            return SCREENSHOT, 0.99

    class FakePipeline:
        def __init__(self, landmarks):
            self.landmarks = landmarks

        def process(self, hands):
            return self.landmarks, {"feature_vector": [0.0] * FEATURE_COUNT}

    engine = GestureEngine.__new__(GestureEngine)
    engine.classifier = FakeClassifier()
    engine.pipeline = FakePipeline(as_landmark_dict(screenshot_pose()))
    valid_pose = engine.process([object()])
    assert valid_pose.label == SCREENSHOT

    engine.pipeline = FakePipeline({"Wrist": {"x": 0.5, "y": 0.5, "z": 0.0}})
    rejected_pose = engine.process([object()])
    assert rejected_pose.label == IDLE


def test_screenshot_is_one_shot_and_releases_drag():
    actions = FakeActions()
    controller = GestureController(actions)
    landmarks = {"Index_MCP": {"x": 0.5, "y": 0.5}}

    controller.update(FIST, landmarks, now=1.0)
    controller.update(SCREENSHOT, landmarks, now=1.1)
    for frame in range(5):
        controller.update(SCREENSHOT, landmarks, now=1.2 + frame * 0.1)

    assert actions.calls.count("screenshot") == 1
    assert actions.calls.count("mouse_up") == 1
    assert "click" not in actions.calls
    assert "right_click" not in actions.calls
    assert controller.last_action[0] == "SCREENSHOT CAPTURED"

    controller.update(IDLE, landmarks, now=2.0)
    controller.update(IDLE, landmarks, now=2.3)
    controller.update(SCREENSHOT, landmarks, now=3.0)
    assert actions.calls.count("screenshot") == 2

    controller.update(SCREENSHOT, None, now=3.1)
    controller.update(SCREENSHOT, None, now=3.6)
    controller.update(SCREENSHOT, landmarks, now=5.0)
    assert actions.calls.count("screenshot") == 3


def test_screenshot_fails_gracefully_and_dry_run_reports_status():
    class FailingActions(FakeActions):
        def screenshot(self):
            raise OSError("screen unavailable")

    controller = GestureController(FailingActions())
    controller.update(SCREENSHOT, {"Index_MCP": {"x": 0.5, "y": 0.5}}, now=1.0)
    assert controller.last_action[0] == "SCREENSHOT FAILED"

    dry_controller = GestureController(DryRunActions(verbose=False))
    dry_controller.update(SCREENSHOT, {"Index_MCP": {"x": 0.5, "y": 0.5}}, now=1.0)
    assert dry_controller.last_action[0] == "SCREENSHOT (DRY RUN)"


def test_existing_controller_gestures_still_dispatch():
    actions = FakeActions()
    controller = GestureController(actions)
    landmarks = {"Index_MCP": {"x": 0.5, "y": 0.5}}

    controller.update(POINTER, landmarks, now=1.0)
    controller.update(LEFT_CLICK, landmarks, now=1.1)
    controller.update(PEACE, landmarks, now=1.5)
    controller.update(FIST, landmarks, now=2.0)
    controller.update(THUMB_UP, landmarks, now=2.3)
    controller.update(THUMB_DOWN, landmarks, now=2.6)

    assert "move" in actions.calls
    assert "click" in actions.calls
    assert "right_click" in actions.calls
    assert "mouse_down" in actions.calls
    assert "mouse_up" in actions.calls
    assert "volume_up" in actions.calls
    assert "volume_down" in actions.calls


def test_system_action_saves_desktop_capture_to_directory(tmp_path, monkeypatch):
    from controller import actions as actions_module
    from controller.actions import SystemActions

    class FakeCapture:
        def size(self):
            return 1920, 1080

        def position(self):
            return 500, 500

        def screenshot(self):
            return Image.new("RGB", (8, 6), "white")

    monkeypatch.setattr(actions_module, "SCREENSHOT_DIR", str(tmp_path / "screenshots"))
    actions = SystemActions.__new__(SystemActions)
    actions._pg = FakeCapture()

    path = actions.screenshot()

    assert (tmp_path / "screenshots").is_dir()
    assert path.startswith(str(tmp_path / "screenshots"))
    assert path.endswith(".png")
    assert Image.open(path).size == (8, 6)