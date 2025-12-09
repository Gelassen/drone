import numpy as np
import pytest
from custom_behavior.optical_flow.optical_flow_v1 import DroneController  


class MockTag:
    def __init__(self, corners, pose_R, pose_t):
        self.corners = corners
        self.pose_R = pose_R
        self.pose_t = pose_t

# @pytest.fixture
# def controller(monkeypatch):
#     # создаем мок детектора
#     class DummyDetector:
#         def __init__(self, *args, **kwargs):
#             pass
#         def detect(self, gray_image, camera_params=None, tag_size=None):
#             return []

#     # подменяем оригинальный класс на мок
#     monkeypatch.setattr(
#         'custom_behavior.optical_flow.optical_flow_v1.AprilTagDetector',
#         DummyDetector
#     )

#     # создаем контроллер
#     return DroneController(disable_mav=True, is_debug=False)


@pytest.fixture
def controller():
    return DroneController(disable_mav=True, is_debug=False)

def test_px_size(controller):
    # квадрат 1x1
    corners = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    px_size = controller._px_size_from_corners(corners)
    assert abs(px_size - 1.0) < 1e-6

def test_pose_height(controller):
    # pose_t = [0,0,2]
    pose_t = np.array([[0],[0],[2]])
    h = controller._pose_height(pose_t)
    assert abs(h - 2.0) < 1e-6

def test_pose_cos_z(controller):
    # identity rotation -> Z axis aligned
    R = np.eye(3)
    cos_z = controller._pose_cos_z(R)
    assert abs(cos_z - 1.0) < 1e-6
    # 45 deg rotation around X -> Z axis tilts
    angle = np.pi/4
    R_x = np.array([[1,0,0],
                    [0,np.cos(angle), -np.sin(angle)],
                    [0,np.sin(angle), np.cos(angle)]])
    cos_z2 = controller._pose_cos_z(R_x)
    assert abs(cos_z2 - np.cos(angle)) < 1e-6

def test_height_from_px(controller):
    px_size = 50
    cos_theta = 1.0
    h = controller._height_from_px(px_size, cos_theta)
    expected = controller.focal_length_px * controller.tag_size_m / px_size
    assert abs(h - expected) < 1e-6

def test_estimate_height(controller):
    # corners square 50x50 pixels, pose_t = [0,0,2], identity rotation
    corners = np.array([[0,0],[50,0],[50,50],[0,50]], dtype=float)
    pose_R = np.eye(3)
    pose_t = np.array([[0],[0],[2]])
    tag = MockTag(corners, pose_R, pose_t)

    h_used, cos_z, raw_px = controller._estimate_height(tag)
    assert h_used is not None
    assert cos_z is not None
    assert raw_px > 0

def test_estimate_height_oblique(controller):
    # rotate around X by 80 deg -> cos_z small -> should skip height
    angle = np.deg2rad(80)
    R_x = np.array([[1,0,0],
                    [0,np.cos(angle), -np.sin(angle)],
                    [0,np.sin(angle), np.cos(angle)]])
    pose_t = np.array([[0],[0],[2]])
    corners = np.array([[0,0],[50,0],[50,50],[0,50]], dtype=float)
    tag = MockTag(corners, R_x, pose_t)

    h_used, cos_z, raw_px = controller._estimate_height(tag)
    # cos_z small -> height_used should be None
    assert cos_z < controller.angle_cos_threshold
    assert h_used is None
