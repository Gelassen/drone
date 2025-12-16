import pytest
import numpy as np
from custom_behavior.optical_flow.april_tag_detector import AprilTagDetector

@pytest.fixture
def detector():
    return AprilTagDetector()

def test_find_squares_returns_list(detector):
    # создаём черное RGB изображение с белым квадратом
    img = np.zeros((200, 200, 3), dtype=np.uint8)  # теперь 3 канала
    img[50:150, 50:150] = [255, 255, 255]
    squares = detector.find_squares(img, min_area=100, max_area=10000)
    assert isinstance(squares, list)
    assert len(squares) > 0

def test_estimate_distance_from_px_positive(detector):
    px_size = 50
    dist = detector.estimate_distance_from_px(px_size)
    assert dist > 0

def test_detect_tags_in_roi_returns_list(detector):
    roi = np.zeros((100, 100, 3), dtype=np.uint8)
    tags = detector.detect_tags_in_roi(roi)
    assert isinstance(tags, list)

def test_find_squares_filters_by_area(detector):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[10:20, 10:20] = 255  # маленький квадрат
    img[50:150, 50:150] = 255  # большой квадрат
    squares = detector.find_squares(img, min_area=200, max_area=10000)
    for x, y, w, h in squares:
        area = w * h
        assert area >= 200 and area <= 10000
