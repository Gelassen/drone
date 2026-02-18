from typing import Dict
from custom_behavior.optical_flow.models.signal_model import Channel, ChannelConfidence
from custom_behavior.optical_flow.models.signal_model import SignalGateThresholds

class SignalGate:

    def __init__(
        self,
        config: SignalGateThresholds
    ):
        self.enable_th = config.enable_threshold
        self.disable_th = config.disable_threshold
        self.min_time = config.min_confidence_time_ms

        # Channel → enabled / disabled
        self.state: Dict[Channel, bool] = {}

        # Channel → timestamp of last state change
        self.last_change_ts: Dict[Channel, int] = {}

    def update(self, channel_conf: ChannelConfidence) -> bool:
        """
        Decide whether a control channel is enabled.

        Returns:
            bool: enabled / disabled
        """

        channel = channel_conf.channel
        confidence = channel_conf.value
        ts = channel_conf.ts

        enabled = self.state.get(channel, False)
        last_ts = self.last_change_ts.get(channel, ts)

        # --- Anti-flutter (confidence must persist in time) ---
        if ts - last_ts < self.min_time:
            return enabled

        # --- Enable channel ---
        if not enabled and confidence >= self.enable_th:
            self.state[channel] = True
            self.last_change_ts[channel] = ts
            return True

        # --- Disable channel ---
        if enabled and confidence <= self.disable_th:
            self.state[channel] = False
            self.last_change_ts[channel] = ts
            return False

        return enabled
