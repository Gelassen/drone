from custom_behavior.optical_flow.models.signal_model import FunctionType

class AdaptiveGainScheduler:

    def __init__(
        self,
        min_conf: float,
        max_conf: float,
        min_gain: float = 0.0,
        max_gain: float = 1.0,
        curve: FunctionType = FunctionType.LINEAR
    ):
        self.min_conf = min_conf
        self.max_conf = max_conf
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.curve = curve

    def gain(self, confidence: float) -> float:
        if confidence <= self.min_conf:
            return self.min_gain

        if confidence >= self.max_conf:
            return self.max_gain

        x = (confidence - self.min_conf) / (self.max_conf - self.min_conf)

        if self.curve == FunctionType.LINEAR:
            return self.min_gain + x * (self.max_gain - self.min_gain)

        elif self.curve == FunctionType.QUADRATIC:
            return self.min_gain + (x ** 2) * (self.max_gain - self.min_gain)

        elif self.curve == FunctionType.SQRT:
            return self.min_gain + (x ** 0.5) * (self.max_gain - self.min_gain)

        else:
            raise ValueError("Unknown curve")
