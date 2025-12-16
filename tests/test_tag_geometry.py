import pytest
import numpy as np
from custom_behavior.optical_flow.tag_geometry import TagGeometry

# ----------------------------
# Тесты для px_size_from_corners
# ----------------------------
def test_px_size_from_corners_square():
    corners = np.array([[0, 0], [0, 2], [2, 2], [2, 0]], dtype=float)
    size = TagGeometry.px_size_from_corners(corners)
    assert pytest.approx(size, 0.01) == 2.0

def test_px_size_from_corners_rect():
    corners = np.array([[0, 0], [0, 2], [4, 2], [4, 0]], dtype=float)
    size = TagGeometry.px_size_from_corners(corners)
    assert pytest.approx(size, 0.01) == 3.0  # (2+4+2+4)/4

# ----------------------------
# Тесты для pose_height
# ----------------------------
def test_pose_height_normal():
    t = np.array([[0], [0], [5]])
    height = TagGeometry.pose_height(t)
    assert height == 5.0

def test_pose_height_invalid():
    t = np.array([[0, 0], [0, 0]])
    assert TagGeometry.pose_height(t) is None

# ----------------------------
# Тесты для pose_cos_z
# ----------------------------
def test_pose_cos_z_normal():
    R = np.eye(3)
    cos_z = TagGeometry.pose_cos_z(R)
    assert cos_z == 1.0

def test_pose_cos_z_invalid():
    R = np.array([[1, 0], [0, 1]])  # некорректная матрица
    assert TagGeometry.pose_cos_z(R) is None

