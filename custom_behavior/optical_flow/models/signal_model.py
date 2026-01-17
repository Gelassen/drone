from enum import Enum
from custom_behavior.optical_flow.target_detection import TargetDetection

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

    def __init__(
        self,
        name: SignalName,
        value: float,
        ts: int
    ):
        self.name=name
        self.value=value
        self.ts=ts

    def __str__(self) -> str:
        return f"Signal(name={self.name}, value={self.value}, ts={self.ts})"

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
    NOISE_STD = "noise_std"
    NOISE_RMS = "noise_rms"
    STABILITY = "stability"
    MONOTONIC = "monotonic"
    DROPOUT_RATE = "dropout"
    LATENCY = "latency"
    SPECTRAL_DENSITY = "spectral density"

class SignalMetrics:
    rms_noise: float 
    std_noise: float 
    spectral_density: float
    sign_stability: float
    monotonic: float
    dropout_rate: float
    latency: float

    def __init__(
        self,
        rms_noise: float,
        std_noise: float,
        spectral_density: float,
        sign_stability: float,
        monotonic: float,
        dropout_rate: float,
        latency: float,
    ) -> None:
        self.rms_noise = rms_noise
        self.std_noise = std_noise
        self.spectral_density = spectral_density
        self.sign_stability = sign_stability
        self.monotonic = monotonic
        self.dropout_rate = dropout_rate
        self.latency = latency


class SignalConfidence:
    signal_name: SignalName
    value: float          # 0..1
    ts: int
    components: dict      # для дебага

    def __init__(
        self,
        signal_name: SignalName,
        value: float,
        ts: int,
        components: dict,
    ) -> None:
        self.signal_name = signal_name
        self.value = value
        self.ts = ts
        self.components = components

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

    def __init__(
        self,
        channel: Channel,
        value: float,
        signals: dict,
        ts: int,
    ) -> None:
        self.channel = channel
        self.value = value
        self.signals = signals
        self.ts = ts


class FunctionType(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    SQRT = "sqrt"

SIGNAL_METRIC_APPLICABILITY = {
    SignalName.MARKER_X_POSITION: {
        SignalMetricsNames.NOISE_STD,
        SignalMetricsNames.STABILITY,
        SignalMetricsNames.MONOTONIC,
        SignalMetricsNames.DROPOUT_RATE,
        SignalMetricsNames.LATENCY,
    },
    SignalName.MARKER_Y_POSITION: {
        SignalMetricsNames.NOISE_STD,
        SignalMetricsNames.STABILITY,
        SignalMetricsNames.MONOTONIC,
        SignalMetricsNames.DROPOUT_RATE,
        SignalMetricsNames.LATENCY,
    },
    SignalName.MARKER_X_AXIS_ANGLE: {
        SignalMetricsNames.NOISE_STD,
        SignalMetricsNames.STABILITY,
        SignalMetricsNames.MONOTONIC,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_Y_AXIS_ANGLE: {
        SignalMetricsNames.NOISE_STD,
        SignalMetricsNames.STABILITY,
        SignalMetricsNames.MONOTONIC,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_ASPECT_RATIO: {
        SignalMetricsNames.NOISE_STD,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_SKEW: {
        SignalMetricsNames.NOISE_STD,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_X_SPEED: {
        SignalMetricsNames.NOISE_RMS,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_Y_SPEED: {
        SignalMetricsNames.NOISE_RMS,
        SignalMetricsNames.DROPOUT_RATE,
    },
    SignalName.MARKER_ROTATION_SPEED: {
        SignalMetricsNames.NOISE_RMS,
        SignalMetricsNames.DROPOUT_RATE,
    },
}

CHANNEL_TO_SIGNALS = {
    Channel.IMAGE_X: [SignalName.MARKER_X_POSITION, SignalName.MARKER_X_SPEED],
    Channel.IMAGE_Y: [SignalName.MARKER_Y_POSITION, SignalName.MARKER_Y_SPEED],
    Channel.ANGLE: [SignalName.MARKER_X_AXIS_ANGLE, SignalName.MARKER_Y_AXIS_ANGLE],
    Channel.OMEGA: [SignalName.MARKER_ROTATION_SPEED],
}

class ManagingCommand:
    velocity_x: float
    velocity_y: float
    velocity_z: float
    yaw: float


