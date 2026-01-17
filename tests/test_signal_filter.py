import pytest

from custom_behavior.optical_flow.signals.signal_filter import SignalFilter
from custom_behavior.optical_flow.models.signal_model import (
    Signal,
    SignalName,
    SignalMetricsNames
)

@pytest.fixture
def signal_filter():
    return SignalFilter()

def test_allows_on_noise_std_signal_for_marker_x_position_returns_true(signal_filter: SignalFilter):
    signal: Signal = Signal(
        name=SignalName.MARKER_X_POSITION,
        value=8.41248614944497,
        ts=1768636437784
    )

    is_allowed: bool = signal_filter.allows(signal=signal, metric=SignalMetricsNames.NOISE_STD)

    assert is_allowed == True