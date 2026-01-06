import time

from signals.signal_adaptive_gain_scheduler import AdaptiveGainScheduler
from signals.signal_arbitrator import Arbitrator
from signals.signal_gate import SignalGate
from signals.signal_confidence import ConfidenceLayer
from signals.signal_evaluation import SignalEvaluator
from signals.signal_utils import SignalsUtil
from signals.signal_filter import SignalFilter

from models.signal_model import (
    Axis,
    Aspect,
    Signal,
    SignalMetricsNames,
    SIGNAL_ALLOWED_METRICS,
    SignalConfidence,
    ChannelConfidence
)
from target_detection import TargetDetection
from converters import converters


class OpticalVelocityControllerV2:

    def __init__(
            self,
            scheduler = AdaptiveGainScheduler(),
            arbitrator = Arbitrator(),
            signal_gate = SignalGate(),
            confidence_layer = ConfidenceLayer(),
            signal_evaluator = SignalEvaluator(),
            signal_util = SignalsUtil(),
            signal_filter = SignalFilter()
    ):
        self.scheduler = scheduler
        self.arbitrator = arbitrator
        self.signal_gate = signal_gate
        self.confidence_layer = confidence_layer
        self.signal_evaluator = signal_evaluator
        self.signal_util = signal_util
        
        self.previous_detection = None

        self.prepare_rms_of_noise = signal_evaluator.prepare_rms_of_noise()
        self.prepare_spectral_density = signal_evaluator.prepare_spectral_density()
        self.prepare_dropout_rate = signal_evaluator.prepare_dropout_rate()
        self.prepare_sign_stability = signal_evaluator.prepare_sign_stability()
        self.prepare_latency = signal_evaluator.prepare_latency()
        self.prepare_monotonic_coefficient = signal_evaluator.prepare_monotonic_coefficient()
        self.signal_filter = signal_filter
    
    def compute(self, detection: TargetDetection, target_alt: int) -> None:
        if self.previous_detection is None:
            print("OpticalVelocityControllerV2::compute - collect 1st marker, skip analysis at this step")
            self.previous_detection = detection
            return None

        axis: Axis = self.signal_util.detect_axis(detection)
        aspect: Aspect = self.signal_util.detect_aspect(detection)
        skew: list = self.signal_util.detect_skew(detection)
        speed: list = self.signal_util.detect_target_speed(detection)
        rotation_speed: float = self.signal_util.detect_rotation_speed(detection)

        current_time_in_ms = time.time() * 1000

        signals = [
            converters.marker_x_position_signal(detection, self.previous_detection),
            converters.marker_y_position_signal(detection, self.previous_detection),
            converters.marker_x_axis_angle_signal(axis, current_time_in_ms),
            converters.marker_y_axis_angle_signal(axis, current_time_in_ms),
            converters.marker_aspect_ratio_signal(aspect, current_time_in_ms),
            converters.marker_width_signal(aspect, current_time_in_ms),
            converters.marker_height_signal(aspect, current_time_in_ms),
            converters.marker_skew_signal(skew, current_time_in_ms),
            converters.marker_x_speed_signal(speed[0], current_time_in_ms),
            converters.marker_y_speed_signal(speed[1], current_time_in_ms),
            converters.marker_rotation_speed_signal(rotation_speed),
        ]

        evaluated = {}

        for signal in signals:
            evaluated[signal.name] = self.evaluate_signal_metrics(signal)

        # --- Confidence ---
        signal_confidences = {}  # SignalName → SignalConfidence
        current_ts = int(time.time() * 1000)
        for signal_name, metrics in evaluated.items():
            conf_value = self.confidence_layer.compute(metrics)
            signal_confidences[signal_name] = SignalConfidence(
                signal_name=signal_name,
                value=conf_value,
                ts=current_ts,
                components=metrics.__dict__  # optional: debug info
            )

        # TODO: migrate to converter
        channel_confidences = {}  # Channel → ChannelConfidence
        for channel, required_signals in signal_confidences.items():
            # CHANNEL_TO_SIGNALS is mapping like:
            # Channel.IMAGE_X → [SignalName.MARKER_X_POSITION, SignalName.MARKER_X_SPEED]
            signals_in_channel = {s: signal_confidences[s] for s in required_signals if s in signal_confidences}
            if signals_in_channel:
                # simplest: take min confidence among required signals
                channel_value = min(s.value for s in signals_in_channel.values())
                channel_confidences[channel] = ChannelConfidence(
                    channel=channel,
                    value=channel_value,
                    signals=signals_in_channel,
                    ts=current_ts
                )

        # --- Gating ---
        gated_channels = {}
        for channel, channel_conf in channel_confidences.items():
            if self.signal_gate.update(channel_conf):
                gated_channels[channel] = channel_conf

        # --- Arbitration ---
        command = self.arbitrator.select(gated_channels)

        # --- Gain scheduling ---
        self.scheduler.gain(confidence, target_alt)

        self.previous_detection = detection
        return command

    def evaluate_signal_metrics(self, signal: Signal) -> dict:
        metrics = {}

        allowed = SIGNAL_ALLOWED_METRICS.get(signal.name, set())

        evaluators = {
            SignalMetricsNames.NOISE: self.prepare_rms_of_noise,
            SignalMetricsNames.SPECTRAL_DENSITY: self.prepare_spectral_density,
            SignalMetricsNames.DROPOUT_RATE: self.prepare_dropout_rate,
            SignalMetricsNames.STABILITY: self.prepare_sign_stability,
            SignalMetricsNames.LATENCY: self.prepare_latency,
            SignalMetricsNames.MONOTONIC: self.prepare_monotonic_coefficient,
        }

        for metric, evaluator in evaluators.items():
            if self.signal_filter.allows(metric):
                metrics[metric] = evaluator(signal)
            else:
                metrics[metric] = None  

        return metrics
