from enum import Enum
from dataclasses import dataclass
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

    def __init__(
        self,
        x_axis: float,
        y_axis: float,
        x_angle: float,
        y_angle: float
    ):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_angle = x_angle
        self.y_angle = y_angle

class Aspect:
    width: float
    height: float
    aspect: float

    def __init__(self, width: float, height: float, aspect: float) -> None:
        self.width = width
        self.height = height
        self.aspect = aspect

class Speed:
    vx: float
    vy: float
    rotation_speed: float  # omega
    prev_marker: TargetDetection

    def __init__(
        self,
        vx: float,
        vy: float,
        rotation_speed: float,
        prev_marker: TargetDetection,
    ) -> None:
        self.vx = vx
        self.vy = vy
        self.rotation_speed = rotation_speed
        self.prev_marker = prev_marker

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

    def to_dict(self) -> dict:
        return {
            "signal": self.signal_name.name,
            "value": self.value,
            "ts": self.ts,
            "components": self._serialize_components(self.components),
        }

    @staticmethod
    def _serialize_components(components: dict) -> dict:
        out = {}
        for k, v in components.items():
            key = k.name if isinstance(k, Enum) else str(k)

            if isinstance(v, Enum):
                out[key] = v.name
            elif hasattr(v, "to_dict"):
                out[key] = v.to_dict()
            elif isinstance(v, (int, float, str, bool)) or v is None:
                out[key] = v
            else:
                out[key] = str(v)   # fallback для дебага
        return out

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

    def to_dict(self) -> dict:
        return {
            "channel": self.channel.name if isinstance(self.channel, Enum) else str(self.channel),
            "value": self.value,
            "ts": self.ts,
            "signals": self._serialize_signals(self.signals),
        }

    @staticmethod
    def _serialize_signals(signals: dict) -> dict:
        out = {}

        for signal_name, confidence in signals.items():
            key = signal_name.name if isinstance(signal_name, Enum) else str(signal_name)

            if hasattr(confidence, "to_dict"):
                out[key] = confidence.to_dict()
            elif isinstance(confidence, (int, float, str, bool)) or confidence is None:
                out[key] = confidence
            else:
                out[key] = str(confidence)  # fallback только для дебага

        return out

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

    def __init__(
        self,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0,
        yaw: float = 0.0
    ) -> None:
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.velocity_z = velocity_z
        self.yaw = yaw

@dataclass(frozen=True)
class ArbitratorThresholds:
    min_image_conf: float
    min_angle_conf: float
    min_omega_conf: float

class ArbitratorConfig:
    """Predefined threshold sets for the Arbitrator."""

    @staticmethod
    def aggressive() -> ArbitratorThresholds:
        return ArbitratorThresholds(
            min_image_conf=0.6,
            min_angle_conf=0.55,
            min_omega_conf=0.5

        )
    
    @staticmethod
    def moderate() -> ArbitratorThresholds:
        return ArbitratorThresholds(
            min_image_conf=0.5,
            min_angle_conf=0.45,
            min_omega_conf=0.4
        )

    @staticmethod
    def loose() -> ArbitratorThresholds:
        return ArbitratorThresholds(
            min_image_conf=0.4,
            min_angle_conf=0.34,
            min_omega_conf=0.3
        )


@dataclass(frozen=True)
class SignalGateThresholds:
    enable_threshold: float
    disable_threshold: float
    min_confidence_time_ms: int

class SignalGateConfig:
    """Predefined threshold sets for SignalGate."""

    @staticmethod
    def aggressive() -> SignalGateThresholds:
        return SignalGateThresholds(
            enable_threshold=0.65,
            disable_threshold=0.45,
            min_confidence_time_ms=100
        )

    @staticmethod
    def moderate() -> SignalGateThresholds:
        return SignalGateThresholds(
            enable_threshold=0.55,
            disable_threshold=0.40,
            min_confidence_time_ms=33
        )

    @staticmethod
    def loose() -> SignalGateThresholds:
        return SignalGateThresholds(
            enable_threshold=0.5,
            disable_threshold=0.3,
            min_confidence_time_ms=25
        )
