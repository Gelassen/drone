import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from custom_behavior.optical_flow.optical_flow_v1 import AprilTagOpticalController
from tests.mock.fake_detection import FakeDetection
from tests.mock.fake_hardware import FakeHardware
from tests.mock.fake_video_source import FakeVideoSource
from tests.mock.fake_detector import FakeDetector


# ----------------------------
# executor fixture
# ----------------------------
@pytest.fixture
def executor():
    executor = ThreadPoolExecutor(max_workers=1)
    yield executor
    executor.shutdown(wait=True, cancel_futures=True)


# ----------------------------
# Один кадр → команда скорости
# ----------------------------
@pytest.mark.asyncio
async def test_single_frame_generates_velocity(executor: ThreadPoolExecutor):
    detections = [FakeDetection(cx=320, cy=240, px_size=50)]
    detector = FakeDetector(detections)

    video = FakeVideoSource([object()])
    hardware = FakeHardware()

    controller = AprilTagOpticalController(
        executor=executor,
        video_source=video,
        hardware=hardware,
        detector=detector,
        lost_frame_threshold=3
    )

    controller._display = Mock()
    controller._draw_debug = Mock()

    await controller.setup()
    await controller._iteration(target_alt=1.5)
    await controller.teardown()

    # assert len(hardware.velocities) == 1
    assert hardware.is_offboard() is False
    assert hardware.is_armed() is False


# ----------------------------
# Потеря цели → landing
# ----------------------------
@pytest.mark.asyncio
async def test_frame_loss_triggers_landing(executor):
    detector = FakeDetector([None, None, None])

    video = FakeVideoSource([object(), object(), object()])
    hardware = FakeHardware()

    controller = AprilTagOpticalController(
        executor=executor,
        video_source=video,
        hardware=hardware,
        detector=detector,
        lost_frame_threshold=2
    )

    controller._display = Mock()
    controller._draw_debug = Mock()

    await controller.setup()

    for _ in range(3):
        await controller._iteration(target_alt=1.5)

    await controller.teardown()

    assert hardware.is_land_called() is True
    assert hardware.is_offboard() is False


# ----------------------------
# Smoke test run()
# ----------------------------
@pytest.mark.asyncio
async def test_run_smoke(executor):
    detector = FakeDetector([
        FakeDetection(cx=320, cy=240, px_size=50),
        None
    ])

    video = FakeVideoSource([object(), object()])
    hardware = FakeHardware()

    controller = AprilTagOpticalController(
        executor=executor,
        video_source=video,
        hardware=hardware,
        detector=detector,
        lost_frame_threshold=1
    )

    controller._display = Mock()
    controller._draw_debug = Mock()

    await controller.run(runtime_sec=0.1)

    assert hardware.is_land_called() is True
    assert hardware.is_offboard() is False
