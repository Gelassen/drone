import numpy as np
import pytest
from unittest.mock import Mock

from custom_behavior.optical_flow.optical_velocity_controller import OpticalVelocityController


class FakeDetection:
    def __init__(self, cx, cy, px_size):
        self.cx = cx
        self.cy = cy
        self.px_size = px_size
