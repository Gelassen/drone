from models.signal_model import (
    Signal,
    SignalMetricsNames,
    SIGNAL_METRIC_APPLICABILITY
)

class SignalFilter:

    def allows(self, signal: Signal, metric: SignalMetricsNames) -> bool:
        allowed = SIGNAL_METRIC_APPLICABILITY.get(signal.name)
        if not allowed:
            return False
        return metric in allowed
