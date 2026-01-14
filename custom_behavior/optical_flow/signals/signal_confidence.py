
from custom_behavior.optical_flow.models.signal_model import (
    SignalMetrics,
    SignalMetricsNames
)

class ConfidenceLayer:

    def __init__(self, weights: dict[str, float]):
        """
        weights: веса компонент, например:
        {
            "noise": 0.3,
            "stability": 0.3,
            "monotonic": 0.2,
            "dropout": 0.1,
            "latency": 0.1
        }
        """
        self.w = weights

    @classmethod
    def with_default_weights(cls):
        return cls(weights={
            SignalMetricsNames.STABILITY:         0.25,
            SignalMetricsNames.NOISE:             0.20,
            SignalMetricsNames.MONOTONIC:         0.15,
            SignalMetricsNames.DROPOUT_RATE:      0.15,
            SignalMetricsNames.LATENCY:           0.15,
            SignalMetricsNames.SPECTRAL_DENSITY:  0.10,
        })

    def compute(self, metrics: SignalMetrics) -> float:
        """
        Возвращает confidence ∈ [0,1]
        """

        scores = []

        # 1. Noise (меньше — лучше)
        if metrics.rms_noise is not None:
            noise_score = max(0.0, 1.0 - metrics.rms_noise)
            scores.append(self.w[SignalMetricsNames.NOISE] * noise_score)

        # 2. Sign stability
        if metrics.sign_stability is not None:
            scores.append(self.w[SignalMetricsNames.STABILITY] * metrics.sign_stability)

        # 3. Monotonicity
        if metrics.monotonic is not None:
            scores.append(self.w[SignalMetricsNames.MONOTONIC] * metrics.monotonic)

        # 4. Dropout (меньше — лучше)
        if metrics.dropout_rate is not None:
            dropout_score = max(0.0, 1.0 - metrics.dropout_rate)
            scores.append(self.w[SignalMetricsNames.DROPOUT_RATE] * dropout_score)

        # 5. Latency (меньше — лучше)
        if metrics.latency is not None:
            latency_score = 1.0 / (1.0 + metrics.latency)
            scores.append(self.w[SignalMetricsNames.LATENCY] * latency_score)

        if not scores:
            return 0.0

        return float(min(1.0, sum(scores)))
