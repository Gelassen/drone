import pytest

from custom_behavior.optical_flow.signals.signal_evaluation import SignalEvaluator
from custom_behavior.optical_flow.models.signal_model import (
    Signal,
    SignalName,
    SignalMetricsNames
)

@pytest.fixture
def signal_evaluator():
    return SignalEvaluator()

def test_noise_std_on_marker_x_signal_does_not_raise_exception(signal_evaluator: SignalEvaluator):
    signal: Signal = Signal(
        name=SignalName.MARKER_X_POSITION,
        value=8.41248614944497,
        ts=1768636437784
    )
    noise_std = signal_evaluator.prepare_noise_std()

    noise_std(signal=signal)

