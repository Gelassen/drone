
from custom_behavior.optical_flow.models.signal_model import (
    SignalMetrics,
    SignalMetricsNames
)

class ConfidenceLayer:

    def __init__(self, weights: dict[SignalMetricsNames, float]):
        self.w = weights

    #    | Метрика      | Формула             
    #    | ------------ | ------------------- 
    #    | NOISE_RMS    | `1 - value`         
    #    | NOISE_STD    | `1 - value`         
    #    | DROPOUT_RATE | `1 - value`         
    #    | STABILITY    | `value`             
    #    | MONOTONIC    | `value`             
    #    | LATENCY      | `1 / (1 + latency)` 


    @classmethod
    def with_default_weights(cls):
        return cls(weights={
            SignalMetricsNames.STABILITY:         0.25,
            SignalMetricsNames.NOISE_RMS:         0.20,
            SignalMetricsNames.NOISE_STD:         0.20,
            SignalMetricsNames.MONOTONIC:         0.15,
            SignalMetricsNames.DROPOUT_RATE:      0.15,
            SignalMetricsNames.LATENCY:           0.15,
            # SignalMetricsNames.SPECTRAL_DENSITY: 0.10  # ← лучше убрать, пока не используется
        })

    def compute(self, metrics: SignalMetrics) -> float:
        if not isinstance(metrics, SignalMetrics):
            raise TypeError(f"Function input argument is not a {SignalMetrics}. Did you pass the right data? {type(metrics)}")
        
        weighted_scores = []

        # Noise RMS
        if metrics.rms_noise is not None:
            self._assert_normalized(metrics.rms_noise, SignalMetricsNames.NOISE_RMS)
            score = 1.0 - metrics.rms_noise
            self._add_score(
                weighted_scores,
                SignalMetricsNames.NOISE_RMS,
                score
            )

        # Noise STD
        if metrics.std_noise is not None:
            self._assert_normalized(metrics.std_noise, SignalMetricsNames.NOISE_STD)
            score = 1.0 - metrics.std_noise
            self._add_score(
                weighted_scores,
                SignalMetricsNames.NOISE_STD,
                score
            )

        if metrics.sign_stability is not None:
            self._add_score(
                weighted_scores,
                SignalMetricsNames.STABILITY,
                metrics.sign_stability
            )

        if metrics.monotonic is not None:
            self._add_score(
                weighted_scores,
                SignalMetricsNames.MONOTONIC,
                metrics.monotonic
            )

        # Dropout
        if metrics.dropout_rate is not None:
            self._assert_normalized(metrics.dropout_rate, SignalMetricsNames.DROPOUT_RATE)
            score = 1.0 - metrics.dropout_rate
            self._add_score(
                weighted_scores,
                SignalMetricsNames.DROPOUT_RATE,
                score
            )

        if metrics.latency is not None:
            score = 1.0 / (1.0 + metrics.latency)
            self._add_score(
                weighted_scores,
                SignalMetricsNames.LATENCY,
                score
            )

        if not weighted_scores:
            return 0.0
        

        total_weight = sum(w for w, _ in weighted_scores)
        
        if total_weight <= 0.0:
            return 0.0
    
        return sum(w * s for w, s in weighted_scores) / total_weight

    def _add_score(self, bucket, metric_name, score):
        weight = self.w.get(metric_name)
        if weight is None:
            return
        bucket.append((weight, score))

    def _assert_normalized(self, value: float, name):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be normalized to [0,1], got {value}")

