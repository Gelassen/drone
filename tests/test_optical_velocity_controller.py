import pytest
from unittest.mock import Mock

from custom_behavior.optical_flow.optical_velocity_controller import OpticalVelocityController
from tests.mock.fake_detector import FakeDetection

@pytest.fixture
def controller():
    detector = Mock()
    detector.get_focal_length_px.return_value = 500
    detector.estimate_distance_from_px.return_value = 10.0

    return OpticalVelocityController(
        detector=detector,
        frame_w=640,
        frame_h=480,
        Kx=1.0,
        Ky=1.0,
        Kz=1.0,
        vx_lim=2.0,
        vy_lim=2.0,
        vz_lim=1.0,
        dead_px=5
    )

def test_centered_target_zero_velocity(controller):
    det = FakeDetection(cx=320, cy=240, px_size=50)

    vx, vy, vz, dist = controller.compute(det, target_alt=10.0)

    assert vx == 0.0
    assert vy == 0.0
    assert vz == 0.0
    assert dist == 10.0

def test_dead_zone_blocks_small_offsets(controller):
    det = FakeDetection(cx=320 + 3, cy=240 - 4, px_size=50)

    vx, vy, vz, _ = controller.compute(det, target_alt=10.0)

    assert vx == 0.0
    assert vy == 0.0

def test_horizontal_offset_generates_vy(controller):
    det = FakeDetection(cx=350, cy=240, px_size=50)

    vx, vy, _, _ = controller.compute(det, target_alt=10.0)

    assert vx == 0.0
    assert vy < 0.0   # цель справа → летим вправо (отрицательный vy)

def test_vertical_offset_generates_vx(controller):
    det = FakeDetection(cx=320, cy=200, px_size=50)

    vx, vy, _, _ = controller.compute(det, target_alt=10.0)

    assert vy == 0.0
    assert vx > 0.0   # цель выше → положительный vx (вперед)


def test_velocity_clipping(controller):
    det = FakeDetection(cx=1000, cy=1000, px_size=1)

    vx, vy, _, _ = controller.compute(det, target_alt=10.0)

    assert abs(vx) == controller.vx_lim
    assert abs(vy) == controller.vy_lim

def test_distance_fallback_to_target_alt(controller):
    controller.detector.estimate_distance_from_px.return_value = None

    det = FakeDetection(cx=320, cy=240, px_size=50)
    _, _, _, dist = controller.compute(det, target_alt=7.5)

    assert dist == 7.5

def test_vertical_velocity(controller):
    controller.detector.estimate_distance_from_px.return_value = 12.0

    det = FakeDetection(cx=320, cy=240, px_size=50)
    _, _, vz, _ = controller.compute(det, target_alt=10.0)

    assert vz > 0.0
