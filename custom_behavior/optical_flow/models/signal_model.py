from enum import Enum
from target_detection import TargetDetection

class SignalName(Enum):
    MARKER_X_POSITION = "cx"          # центр маркера по X
    MARKER_Y_POSITION = "cy"          # центр маркера по Y
    MARKER_X_AXIS_ANGLE = "X"         # угол X оси маркера
    MARKER_Y_AXIS_ANGLE = "Y"         # угол Y оси маркера
    MARKER_ASPECT_RATIO = "aspect_ratio"
    MARKER_WIDTH = "width"
    MARKER_HEIGHT = "height"
    MARKER_SKEW = "skew"
    MARKER_X_SPEED = "V_X"            # скорость по X
    MARKER_Y_SPEED = "V_Y"            # скорость по Y
    MARKER_ROTATION_SPEED = "Omega"   # угловая скорость

class Signal: 
    name: SignalName
    value: float
    ts: int

class Point:

    def __init__(self, x, y):
        self.x = x 
        self.y = y

class Axis:
    x_axis: float
    y_axis: float
    x_angle: float
    y_angle: float


class Aspect:
    width: float
    height: float
    aspect: float

class Speed:
    vx: float 
    vy: float
    rotation_speed: float # omega
    prev_marker: TargetDetection