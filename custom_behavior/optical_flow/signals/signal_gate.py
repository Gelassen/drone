from typing import Dict

from models.signal_model import SignalMetricsNames

class SignalGate:

    def __init__(
        self,
        enable_threshold: float = 0.65,
        disable_threshold: float = 0.45,
        min_confidence_time_ms: int = 100
    ):
        self.enable_th = enable_threshold
        self.disable_th = disable_threshold
        self.min_time = min_confidence_time_ms

        self.state: Dict[SignalMetricsNames, bool] = {}
        self.last_change_ts: Dict[SignalMetricsNames, int] = {}

    def update(self, channel: SignalMetricsNames, confidence: float, ts: int) -> bool:
        """
        Возвращает: enabled / disabled
        """

        enabled = self.state.get(channel, False)
        last_ts = self.last_change_ts.get(channel, ts)

        # Защита от флаттера
        if ts - last_ts < self.min_time:
            return enabled

        if not enabled and confidence >= self.enable_th:
            self.state[channel] = True
            self.last_change_ts[channel] = ts
            return True

        if enabled and confidence <= self.disable_th:
            self.state[channel] = False
            self.last_change_ts[channel] = ts
            return False

        return enabled
