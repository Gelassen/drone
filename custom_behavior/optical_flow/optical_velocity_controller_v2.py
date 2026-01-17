import time

from custom_behavior.optical_flow.signals.signal_adaptive_gain_scheduler import AdaptiveGainScheduler
from custom_behavior.optical_flow.signals.signal_arbitrator import Arbitrator
from custom_behavior.optical_flow.signals.signal_gate import SignalGate
from custom_behavior.optical_flow.signals.signal_confidence import ConfidenceLayer
from custom_behavior.optical_flow.signals.signal_evaluation import SignalEvaluator
from custom_behavior.optical_flow.signals.signal_utils import SignalsUtil
from custom_behavior.optical_flow.signals.signal_filter import SignalFilter
from custom_behavior.optical_flow.signals.signal_buffer import SignalBuffer
from custom_behavior.optical_flow.signals.signal_command_assembler import CommandAssembler

from custom_behavior.optical_flow.models.signal_model import (
    Axis,
    Aspect,
    Signal,
    SignalMetricsNames,
    SignalConfidence,
    ChannelConfidence,
    CHANNEL_TO_SIGNALS,
    Channel,
    SignalName,
    ManagingCommand,
    SignalMetrics
)
from custom_behavior.optical_flow.target_detection import TargetDetection
from custom_behavior.optical_flow.converters import converters


class OpticalVelocityControllerV2:

    RISKS_WEIGHTS = {

    }

    def __init__(
            self,
            scheduler = AdaptiveGainScheduler(min_conf=0.35, max_conf=0.75), # TODO: tune me
            arbitrator = Arbitrator(),
            signal_gate = SignalGate(),
            confidence_layer = ConfidenceLayer.with_default_weights(),
            signal_evaluator = SignalEvaluator(SignalBuffer()),
            signal_util = SignalsUtil(),
            signal_filter = SignalFilter(),
            command_assembler = CommandAssembler()
    ):
        self.scheduler = scheduler
        self.arbitrator = arbitrator
        self.signal_gate = signal_gate
        self.confidence_layer = confidence_layer
        self.signal_evaluator = signal_evaluator
        self.signal_util = signal_util
        
        self.previous_detection = None

        self.prepare_rms_of_noise = signal_evaluator.prepare_noise_rms()
        self.prepare_std_of_noise = signal_evaluator.prepare_noise_std()
        self.prepare_spectral_density = signal_evaluator.prepare_spectral_density()
        self.prepare_dropout_rate = signal_evaluator.prepare_dropout_rate(expected_dt=16.67)
        self.prepare_sign_stability = signal_evaluator.prepare_sign_stability()
        self.prepare_latency = signal_evaluator.prepare_latency()
        self.prepare_monotonic_coefficient = signal_evaluator.prepare_monotonic_coefficient()
        self.signal_filter = signal_filter
        self.command_assembler = command_assembler
    
    def compute(self, detection: TargetDetection, target_alt: int) -> dict:
        if self.previous_detection is None:
            print("OpticalVelocityControllerV2::compute - collect 1st marker, skip analysis at this step")
            self.previous_detection = detection
            return {}

        print("Detection", detection)

        # --- Detect features ---
        axis: Axis = self.signal_util.detect_axis(detection)
        aspect: Aspect = self.signal_util.detect_aspect(detection)
        skew: list = self.signal_util.detect_skew(detection)
        speed: list = self.signal_util.detect_target_speed(
            prev_marker=self.previous_detection, 
            marker=detection
        )
        rotation_speed: float = self.signal_util.detect_rotation_speed(
            prev_marker=self.previous_detection,
            marker=detection
        )

        current_time_in_ms = int(time.time() * 1000)

        print("Detection before signal conversion: ", detection)

        # --- Convert raw detection to signals ---
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
            converters.marker_rotation_speed_signal(rotation_speed, current_time_in_ms),
        ]

        for s in signals:
            print("Signal:", s)
        
        signals_dict = {
            s.name: s
            for s in signals
            if s is not None
        }


        # for s_name, s in signals_dict.items():
            # print("Signal-filtered:", s)
        # --- Evaluate metrics ---
        evaluated = {}
        for signal_name, signal in signals_dict.items():
            evaluated[signal_name] = self.evaluate_signal_metrics(signal)

        print("Evaluated metrics", evaluated)
        evaluated_metrics: dict[SignalName, SignalMetricsNames] = converters.evaluated_metrics_to_signal_metrics(evaluated)

        # --- Compute signal-level confidence ---
        signal_confidences: dict = {}
        for signal_name, metrics in evaluated_metrics.items():
            conf_value = self.confidence_layer.compute(metrics)
            signal_confidences[signal_name] = SignalConfidence(
                signal_name=signal_name,
                value=conf_value,
                ts=current_time_in_ms,
                components=metrics.__dict__  # optional: debug info
            )

        # --- Compute channel-level confidence ---
        channel_confidences: dict = {}
        for channel, required_signals in CHANNEL_TO_SIGNALS.items():
            present_signals = {s: signal_confidences[s] for s in required_signals if s in signal_confidences}
            if present_signals:
                min_conf = min(s.value for s in present_signals.values())
                channel_confidences[channel] = ChannelConfidence(
                    channel=channel,
                    value=min_conf,
                    signals=present_signals,
                    ts=current_time_in_ms
                )

        # --- Gating ---
        gated_channels: dict = {}
        for channel, ch_conf in channel_confidences.items():
            if self.signal_gate.update(channel_conf=ch_conf):
                gated_channels[channel] = ch_conf

        # --- Arbitration ---
        command = self.arbitrator.select(gated_channels)

        # --- Generate raw command values per channel ---
        raw_command = {
            Channel.IMAGE_X: signals_dict[SignalName.MARKER_X_POSITION].value,
            Channel.IMAGE_Y: signals_dict[SignalName.MARKER_Y_POSITION].value,
            Channel.ANGLE: signals_dict[SignalName.MARKER_X_AXIS_ANGLE].value,
            Channel.OMEGA: signals_dict[SignalName.MARKER_ROTATION_SPEED].value,
        }

        # --- Apply adaptive gain ---
        scaled_commands: dict = {}
        if command:
            if isinstance(command, tuple):
                # e.g., IMAGE_X & IMAGE_Y
                for ch in command:
                    gain = self.scheduler.gain(gated_channels[ch].value)
                    scaled_commands[ch] = raw_command[ch] * gain
            else:
                gain = self.scheduler.gain(gated_channels[command].value)
                scaled_commands[command] = raw_command[command] * gain

        # --- Update previous detection ---
        self.previous_detection = detection

        managing_command: ManagingCommand = self.command_assembler.signals_to_command(scaled_commands)

        return managing_command

    def evaluate_signal_metrics(self, signal: Signal) -> dict:
        metrics = {}

        evaluators = {
            SignalMetricsNames.NOISE_STD: self.prepare_std_of_noise,
            SignalMetricsNames.NOISE_RMS: self.prepare_rms_of_noise,
            SignalMetricsNames.SPECTRAL_DENSITY: self.prepare_spectral_density,
            SignalMetricsNames.DROPOUT_RATE: self.prepare_dropout_rate,
            SignalMetricsNames.STABILITY: self.prepare_sign_stability,
            SignalMetricsNames.LATENCY: self.prepare_latency,
            SignalMetricsNames.MONOTONIC: self.prepare_monotonic_coefficient,
        }

        for metric, evaluator in evaluators.items():
            if self.signal_filter.allows(signal, metric):
                metrics[metric] = evaluator(signal)
            else:
                metrics[metric] = None  

        return metrics

