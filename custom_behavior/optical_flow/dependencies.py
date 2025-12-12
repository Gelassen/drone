# dependencies.py
from .april_tag_detector import AprilTagDetector
from .drone_hardware import MAVSDKHardware  # реальный hardware интерфейс

def provide_detector():
    """Возвращает детектор AprilTag"""
    return AprilTagDetector()

def provide_hardware():
    """Возвращает реализацию интерфейса аппаратуры (MAVSDK)"""
    return MAVSDKHardware()
