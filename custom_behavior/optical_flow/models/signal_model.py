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

class SignalMetricsNames(Enum):
    NOISE = "noise"
    STABILITY = "stability"
    MONOTONIC = "monotonic"
    DROPOUT_RATE = "dropout"
    LATENCY = "latency"
    SPECTRAL_DENSITY = "spectral density"

class SignalMetrics:
    rms_noise: float
    spectral_density: float
    sign_stability: float
    monotonic: float
    dropout_rate: float
    latency: float

class SignalConfidence:
    signal_name: SignalName
    value: float          # 0..1
    ts: int
    components: dict      # для дебага

class Channel(Enum):
    IMAGE_X = "image_x"      # cx → roll
    IMAGE_Y = "image_y"      # cy → pitch
    ANGLE   = "angle"        # marker axis angle
    OMEGA   = "omega"        # rotation speed

class ChannelConfidence:
    channel: Channel
    value: float          # 0..1
    signals: dict         # SignalName → SignalConfidence
    ts: int

class FunctionType(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    SQRT = "sqrt"

SIGNAL_METRIC_APPLICABILITY = {
    SignalName.MARKER_X_POSITION: {
        SignalMetricsNames.NOISE,
        SignalMetricsNames.STABILITY,
        SignalMetricsNames.MONOTONIC,
        SignalMetricsNames.DROPOUT_RATE,
        SignalMetricsNames.LATENCY,
    },
    SignalName.MARKER_Y_POSITION: {
        SignalMetricsNames.NOISE,
        SignalMetricsNames.STABILITY,
        SignalMetricsNames.MONOTONIC,
        SignalMetricsNames.DROPOUT_RATE,
        SignalMetricsNames.LATENCY,
    },
    SignalName.MARKER_X_AXIS_ANGLE: {
        SignalMetricsNames.NOISE,
        SignalMetricsNames.STABILITY,
        SignalMetricsNames.MONOTONIC,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_Y_AXIS_ANGLE: {
        SignalMetricsNames.NOISE,
        SignalMetricsNames.STABILITY,
        SignalMetricsNames.MONOTONIC,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_ASPECT_RATIO: {
        SignalMetricsNames.NOISE,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_SKEW: {
        SignalMetricsNames.NOISE,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_X_SPEED: {
        SignalMetricsNames.NOISE,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_Y_SPEED: {
        SignalMetricsNames.NOISE,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_ROTATION_SPEED: {
        SignalMetricsNames.NOISE,
        SignalMetricsNames.DROPOUT_RATE,
    },
}


