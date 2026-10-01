"""Geometry-based recognition for the three-finger screenshot pose."""
import math


_FINGER_JOINTS = {
    "Index": ("Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip"),
    "Middle": ("Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip"),
    "Ring": ("Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip"),
    "Pinky": ("Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip"),
}


def _point(landmarks, name):
    point = landmarks.get(name)
    if not isinstance(point, dict):
        return None
    try:
        values = tuple(float(point[axis]) for axis in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _distance(first, second):
    return math.dist(first, second)


def _angle(first, vertex, last):
    first_vector = tuple(a - b for a, b in zip(first, vertex))
    last_vector = tuple(a - b for a, b in zip(last, vertex))
    first_length = math.sqrt(sum(value * value for value in first_vector))
    last_length = math.sqrt(sum(value * value for value in last_vector))
    if first_length == 0 or last_length == 0:
        return None
    cosine = sum(a * b for a, b in zip(first_vector, last_vector)) / (first_length * last_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _finger_state(landmarks, names):
    points = [_point(landmarks, name) for name in names]
    if any(point is None for point in points):
        return None
    mcp, pip, dip, tip = points
    pip_angle = _angle(mcp, pip, dip)
    dip_angle = _angle(pip, dip, tip)
    segment_length = _distance(mcp, pip)
    if pip_angle is None or dip_angle is None or segment_length == 0:
        return None
    extended = (
        pip_angle >= 150
        and dip_angle >= 150
        and _distance(mcp, tip) / segment_length >= 1.7
    )
    folded = pip_angle <= 135 or dip_angle <= 135
    return extended, folded, mcp, tip


def is_screenshot_pose(landmarks):
    """Reject uncertain data; recognize three extended fingers and two folded/down."""
    if not isinstance(landmarks, dict):
        return False

    wrist = _point(landmarks, "Wrist")
    middle_mcp = _point(landmarks, "Middle_MCP")
    if wrist is None or middle_mcp is None:
        return False
    palm_axis = tuple(b - a for a, b in zip(wrist, middle_mcp))
    palm_scale = math.sqrt(sum(value * value for value in palm_axis))
    if palm_scale == 0:
        return False

    states = {finger: _finger_state(landmarks, joints) for finger, joints in _FINGER_JOINTS.items()}
    if any(state is None for state in states.values()):
        return False
    if not all(states[finger][0] for finger in ("Index", "Middle", "Ring")):
        return False

    pinky_extended, pinky_folded, pinky_mcp, pinky_tip = states["Pinky"]
    pinky_projection = sum(
        (tip - base) * axis for tip, base, axis in zip(pinky_tip, pinky_mcp, palm_axis)
    )
    pinky_down = pinky_extended and pinky_projection < -0.1 * palm_scale * palm_scale
    if not pinky_folded and not pinky_down:
        return False

    thumb_points = [_point(landmarks, name) for name in (
        "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip"
    )]
    if any(point is None for point in thumb_points):
        return False
    thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = thumb_points
    thumb_mcp_angle = _angle(thumb_cmc, thumb_mcp, thumb_ip)
    thumb_ip_angle = _angle(thumb_mcp, thumb_ip, thumb_tip)
    if thumb_mcp_angle is None or thumb_ip_angle is None:
        return False

    palm_points = [wrist]
    for name in ("Index_MCP", "Middle_MCP", "Ring_MCP", "Pinky_MCP"):
        point = _point(landmarks, name)
        if point is None:
            return False
        palm_points.append(point)
    palm_center = tuple(sum(point[axis] for point in palm_points) / len(palm_points) for axis in range(3))
    thumb_folded = (
        thumb_mcp_angle <= 135
        or thumb_ip_angle <= 135
        or _distance(thumb_tip, palm_center) <= 0.85 * palm_scale
    )
    thumb_direction = tuple(tip - base for tip, base in zip(thumb_tip, thumb_mcp))
    thumb_projection = sum(direction * axis for direction, axis in zip(thumb_direction, palm_axis))
    thumb_down = (
        thumb_mcp_angle >= 150
        and thumb_ip_angle >= 150
        and _distance(thumb_mcp, thumb_tip) >= 1.5 * _distance(thumb_mcp, thumb_ip)
        and thumb_projection < -0.1 * palm_scale * palm_scale
    )
    return thumb_folded or thumb_down