import time
import json

from enum import Enum

from custom_behavior.optical_flow.signals.signal_adaptive_gain_scheduler import AdaptiveGainScheduler
from custom_behavior.optical_flow.signals.signal_arbitrator import Arbitrator
from custom_behavior.optical_flow.signals.signal_gate import SignalGate
from custom_behavior.optical_flow.signals.signal_confidence import ConfidenceLayer
from custom_behavior.optical_flow.signals.signal_evaluation import SignalEvaluator
from custom_behavior.optical_flow.signals.signal_utils import SignalsUtil
from custom_behavior.optical_flow.signals.signal_filter import SignalFilter
from custom_behavior.optical_flow.signals.signal_buffer import SignalBuffer
from custom_behavior.optical_flow.signals.signal_command_assembler import CommandAssembler
from custom_behavior.optical_flow.models.models import TelemetryEvents


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
    SignalMetrics,
    ArbitratorThresholds,
    ArbitratorConfig,
    SignalGateConfig
)
from custom_behavior.optical_flow.target_detection import TargetDetection
from custom_behavior.optical_flow.converters import converters
from custom_behavior.utils.telemetry_logger import telemetry

class OpticalVelocityControllerV2:

    RISKS_WEIGHTS = {

    }

    def __init__(
            self,
            scheduler = AdaptiveGainScheduler(min_conf=0.35, max_conf=0.75), # TODO: tune me
            arbitrator = Arbitrator(config=ArbitratorConfig.loose()),
            signal_gate = SignalGate(config=SignalGateConfig.loose()),
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
        
        self.previous_detection: TargetDetection = None

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
        
        telemetry.emit(
            event=TelemetryEvents.APRIL_TAG_DETECTION.name,
            cx=detection.cx,
            cy=detection.cy,
            px_size=detection.px_size,
            source=detection.source,
            side=detection.side,
            corners=detection.corners.tolist() if detection.corners is not None else None,
            homography=detection.homography.tolist() if detection.homography is not None else None,
            timestamp=detection.timestamp
        )

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

        signals_dict = {
            s.name: s
            for s in signals
            if s is not None
        }

        # --- Evaluate metrics ---
        evaluated = {}
        for signal_name, signal in signals_dict.items():
            evaluated[signal_name] = self.evaluate_signal_metrics(signal)

        evaluated_metrics: dict[SignalName, SignalMetricsNames] = converters.evaluated_metrics_to_signal_metrics(evaluated)

        # print("evaluated_metrics", evaluated_metrics)
        self._debug_log_evaluated_metrics(evaluated_metrics)

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
        # print("signal_confidences", signal_confidences)
        self._debug_log_signal_confidences(signal_confidences)

        # --- Compute channel-level confidence ---
        channel_confidences: dict[SignalMetricsNames, ChannelConfidence] = {}
        for channel, required_signals in CHANNEL_TO_SIGNALS.items():
            present_signals = {s: signal_confidences[s] for s in required_signals if s in signal_confidences}
            print("present_signals", present_signals)
            if present_signals:
                min_conf = min(s.value for s in present_signals.values())
                channel_confidences[channel] = ChannelConfidence(
                    channel=channel,
                    value=min_conf,
                    signals=present_signals,
                    ts=current_time_in_ms
                )
        print("channel_confidences", channel_confidences)
        self._debug_log_channel_confidences(channel_confidences)

        # --- Gating ---
        gated_channels: dict = {}
        for channel, ch_conf in channel_confidences.items():
            if self.signal_gate.update(channel_conf=ch_conf):
                gated_channels[channel] = ch_conf

        # print("gated_channels", gated_channels)
        self._debug_log_gated_channel(gated_channels)

        # --- Arbitration ---
        command = self.arbitrator.select(gated_channels)

        self._debug_arbitrator_decision(command)

        # --- Generate raw command values per channel ---
        raw_command = {
            Channel.IMAGE_X: signals_dict[SignalName.MARKER_X_POSITION].value,
            Channel.IMAGE_Y: signals_dict[SignalName.MARKER_Y_POSITION].value,
            # Channel.ANGLE: signals_dict[SignalName.MARKER_X_AXIS_ANGLE].value, # TODO: compensate gain in this edge case
            # Channel.OMEGA: signals_dict[SignalName.MARKER_ROTATION_SPEED].value, # TODO: compensate gain in this edge case
        }

        angle_signal = signals_dict.get(SignalName.MARKER_X_AXIS_ANGLE)
        omega_signal = signals_dict.get(SignalName.MARKER_ROTATION_SPEED)

        if angle_signal is not None:
            raw_command[Channel.ANGLE] = angle_signal.value

        if omega_signal is not None:
            raw_command[Channel.OMEGA] = omega_signal.value

        print("raw commands", raw_command)
        self._debug_log_raw_command(raw_command)

        # TODO: update gain
        # if ANGLE is missing:
            # increase gain on IMAGE_X, IMAGE_Y
            # reduce aggressiveness of OMEGA

        # --- Apply adaptive gain ---
        scaled_commands: dict = {}
        if command:
            print("raw_command, part I")
            if isinstance(command, tuple):
                print("raw_command, part II", type(command))
                # e.g., IMAGE_X & IMAGE_Y
                for ch in command:
                    gain = self.scheduler.gain(gated_channels[ch].value)
                    print("raw_command[ch] ", raw_command[ch])
                    print("gain", gain)
                    self._debug_log_raw_command_gain(ch, raw_command[ch], gain, "type-I")
                    scaled_commands[ch] = raw_command[ch] * gain
                    self._debug_log_raw_command_gain(ch, raw_command[ch], gain, "both")
                    self._debug_log_scale_debug(
                        channel=ch,
                        raw_command_value = raw_command[ch],
                        scaled_commands = {ch: scaled_commands[ch]},
                        gain=gain
                    )
            else:
                ch = command
                gain = self.scheduler.gain(gated_channels[command].value)
                print("raw_command[command] ", raw_command[command])
                print("gain", gain)
                self._debug_log_raw_command_gain(ch, raw_command[ch], gain, "type-II")
                scaled_commands[command] = raw_command[command] * gain
                self._debug_log_raw_command_gain(ch, raw_command[ch], gain, "single")
                self._debug_log_scale_debug(
                        channel=ch,
                        raw_command_value = raw_command[ch],
                        scaled_commands = {ch: scaled_commands[ch]},
                        gain=gain
                    )
        else:
            print("No command has been found!!!")

        print("scaled_commands", scaled_commands)
        self._debug_log_scaled_command(scaled_commands)

        # --- Update previous detection ---
        self.previous_detection = detection

        managing_command: ManagingCommand = self.command_assembler.signals_to_command(scaled_commands)

        telemetry.emit(
            event=TelemetryEvents.MANAGING_COMMAND.name,
            velocity_x=managing_command.velocity_x,
            velocity_y=managing_command.velocity_y,
            velocity_z=managing_command.velocity_z,
            yaw=managing_command.yaw
        )

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

    def _debug_log_evaluated_metrics(self, metrics: dict[SignalName, SignalMetricsNames]):
        for signal_name, signal_metrics in metrics.items():
            telemetry.emit(
                event=TelemetryEvents.SIGNAL_METRICS.name,
                signal_name=signal_name.value,
                rms_noise=signal_metrics.rms_noise, 
                std_noise=signal_metrics.std_noise, 
                spectral_density=signal_metrics.spectral_density,
                sign_stability=signal_metrics.sign_stability,
                monotonic=signal_metrics.monotonic,
                dropout_rate=signal_metrics.dropout_rate,
                latency=signal_metrics.latency
            )

    def _debug_log_signal_confidences(self, signal_conf):
        for signal_name, confidences in signal_conf.items():
            telemetry.emit(
                event=TelemetryEvents.SIGNAL_CONFIDENCE.name,
                signal_name=signal_name.value,
                value=confidences.value,
                ts=confidences.ts,
                components=json.dumps(confidences.components)
            )

    def _debug_log_channel_confidences(self, channel_conf: dict[SignalMetricsNames, ChannelConfidence]):
        for signal_name, channel_confidences in channel_conf.items():
            # print("channel_confidences.signals", channel_confidences)
            # for key, item in channel_confidences.signals.items():
                # print(key.name if isinstance(key, Enum) else key, item)
            serialized_signals = converters.confidence_signals_to_serialized_signals(channel_confidences.signals)
            
            telemetry.emit(
                event=TelemetryEvents.CHANNEL_CONFIDENCE.name,
                channel=signal_name.value,
                value=channel_confidences.value,
                signals=json.dumps(serialized_signals),
                ts=channel_confidences.ts
            )

    def _debug_log_gated_channel(self, gated_channels: dict[Channel, ChannelConfidence]):
        for channel, channel_confidence in gated_channels.items():   # ← .items() !
            telemetry.emit(
                event=TelemetryEvents.GATED_CHANNEL_CONFIDENCE.name,
                channel=channel.name,
                channel_confidence=json.dumps(channel_confidence.to_dict())
            )
            
  
    def _debug_log_raw_command(self, raw_command: dict[Channel, any]):
        for channel, data in raw_command.items():
            telemetry.emit(
                event=TelemetryEvents.RAW_COMMAND.name,
                channel=channel.name,
                value=data
            )

    def _debug_log_raw_command_gain(self, 
                                    channel: Channel, 
                                    value: any, 
                                    gain: any,
                                    type: str
                                ):
        telemetry.emit(
            event=TelemetryEvents.RAW_COMMAND_GAIN.name,
            channel=channel.name,
            value=value,
            type=type,
            gain=gain
        )

    def _debug_log_scaled_command(self, scaled_commands: dict[Channel, any]):
        for channel, command_value in scaled_commands.items():
            telemetry.emit(
                event=TelemetryEvents.SCALED_COMMAND.name,
                channel=channel.name,
                value=command_value
            )

    def _debug_log_scale_debug(self,
        channel: Channel,
        raw_command_value: any, 
        scaled_commands: dict[Channel, any],
        gain: any
    ):
        scaled_value = scaled_commands[channel]

        telemetry.emit(
            event=TelemetryEvents.SCALE_DEBUG.name,
            channel=channel.name,
            raw=raw_command_value,
            gain=gain,
            scaled=scaled_value,
        )

    # def _debug_arbitrator_decision(self, command: Channel):
    #     telemetry.emit(
    #         event=TelemetryEvents.ARBITRATOR_DECISION_COMMAND.name,
    #         command=command.name if command else "None"
    #     )

    def _debug_arbitrator_decision(self, command: Channel | tuple[Channel, Channel] | None) -> None:
        if command is None:
            telemetry.emit(
                event=TelemetryEvents.ARBITRATOR_DECISION_COMMAND.name,
                command="None"
            )
        elif isinstance(command, tuple):
            # предполагаем, что это всегда пара (IMAGE_X, IMAGE_Y)
            if len(command) == 2 and all(isinstance(c, Channel) for c in command):
                telemetry.emit(
                    event=TelemetryEvents.ARBITRATOR_DECISION_COMMAND.name,
                    command="BOTH",
                    channels=[command[0].name, command[1].name]
                    # или command1=command[0].name, command2=command[1].name — как тебе удобнее
                )
            else:
                telemetry.emit(
                    event=TelemetryEvents.ARBITRATOR_DECISION_COMMAND.name,
                    command="INVALID_TUPLE",
                    length=len(command)
                )
        else:
            # одиночный канал
            telemetry.emit(
                event=TelemetryEvents.ARBITRATOR_DECISION_COMMAND.name,
                command=command.name
            )
