import pytest
import asyncio
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

from custom_behavior.optical_flow.video_source import AsyncVideoSource
from tests.mock.fake_capture import FakeCapture

# ----------------------------
# Фикстура executor
# ----------------------------
@pytest.fixture
def executor():
    with ThreadPoolExecutor(max_workers=1) as ex:
        yield ex



# ----------------------------
# Тест открытия
# ----------------------------
@pytest.mark.asyncio
async def test_open_sets_frame_size(executor):
    source = AsyncVideoSource("fake.mp4", executor)

    with patch("cv2.VideoCapture", return_value=FakeCapture()) as mock_cap:
        await source.open()
        assert source.cap is not None
        assert source.frame_w == 640
        assert source.frame_h == 480
        mock_cap.assert_called_once_with("fake.mp4")

# ----------------------------
# Тест чтения кадра
# ----------------------------
@pytest.mark.asyncio
async def test_read_returns_frame(executor):
    fake_cap = FakeCapture(ret=True, frame="my_frame")
    source = AsyncVideoSource("fake.mp4", executor)

    with patch("cv2.VideoCapture", return_value=fake_cap):
        await source.open()
        frame = await source.read()
        assert frame == "my_frame"

# ----------------------------
# Тест чтения None, если ret=False
# ----------------------------
@pytest.mark.asyncio
async def test_read_returns_none_on_failure(executor):
    fake_cap = FakeCapture(ret=False)
    source = AsyncVideoSource("fake.mp4", executor)

    with patch("cv2.VideoCapture", return_value=fake_cap):
        await source.open()
        frame = await source.read()
        assert frame is None

# ----------------------------
# Тест close
# ----------------------------
def test_close_releases_capture(executor):
    source = AsyncVideoSource("fake.mp4", executor)
    fake_cap = FakeCapture()
    source.cap = fake_cap

    source.close()
    assert fake_cap.released is True

# ----------------------------
# Тест ошибки при невозможности открыть
# ----------------------------
@pytest.mark.asyncio
async def test_open_raises_on_fail(executor):
    fake_cap = FakeCapture()
    fake_cap.isOpened = lambda: False

    source = AsyncVideoSource("fake.mp4", executor)

    with patch("cv2.VideoCapture", return_value=fake_cap):
        with pytest.raises(RuntimeError, match="Cannot open video source"):
            await source.open()
