from custom_behavior.optical_flow.target_detection import TargetDetection
from custom_behavior.optical_flow.models.signal_model import (
    Signal,
    SignalName
)

# ---------- центр маркера ----------
def marker_x_position_signal(marker: TargetDetection, prev_marker: TargetDetection) -> Signal:
    if marker is None:
        return None
    return Signal(
        name=SignalName.MARKER_X_POSITION,
        value=float(marker.cx - prev_marker.cx),
        ts=marker.ts
    )

def marker_y_position_signal(marker: TargetDetection, prev_marker: TargetDetection) -> Signal:
    if marker is None:
        return None
    return Signal(
        name=SignalName.MARKER_Y_POSITION,
        value=float(marker.cy - prev_marker.cy),
        ts=marker.ts
    )

# ---------- оси ----------
def marker_x_axis_angle_signal(axis, ts) -> Signal:
    if axis is None:
        return None
    return Signal(
        name=SignalName.MARKER_X_AXIS_ANGLE,
        value=float(axis.x_angle),
        ts=ts
    )

def marker_y_axis_angle_signal(axis, ts) -> Signal:
    if axis is None:
        return None
    return Signal(
        name=SignalName.MARKER_Y_AXIS_ANGLE,
        value=float(axis.y_angle),
        ts=ts
    )

# ---------- аспект ----------
def marker_aspect_ratio_signal(aspect, ts) -> Signal:
    if aspect is None:
        return None
    return Signal(
        name=SignalName.MARKER_ASPECT_RATIO,
        value=float(aspect.aspect),
        ts=ts
    )

def marker_width_signal(aspect, ts) -> Signal:
    if aspect is None:
        return None
    return Signal(
        name=SignalName.MARKER_WIDTH,
        value=float(aspect.width),
        ts=ts
    )

def marker_height_signal(aspect, ts) -> Signal:
    if aspect is None:
        return None
    return Signal(
        name=SignalName.MARKER_HEIGHT,
        value=float(aspect.height),
        ts=ts
    )

# ---------- skew ----------
def marker_skew_signal(skew, ts) -> Signal:
    if skew is None:
        return None
    return Signal(
        name=SignalName.MARKER_SKEW,
        value=float(skew),
        ts=ts
    )

# ---------- скорости ----------
def marker_x_speed_signal(vx, ts) -> Signal:
    if vx is None:
        return None
    return Signal(
        name=SignalName.MARKER_X_SPEED,
        value=float(vx),
        ts=ts
    )

def marker_y_speed_signal(vy, ts) -> Signal:
    if vy is None:
        return None
    return Signal(
        name=SignalName.MARKER_Y_SPEED,
        value=float(vy),
        ts=ts
    )

def marker_rotation_speed_signal(omega, ts) -> Signal:
    if omega is None:
        return None
    return Signal(
        name=SignalName.MARKER_ROTATION_SPEED,
        value=float(omega),
        ts=ts
    )
