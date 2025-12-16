import pytest
from unittest.mock import Mock
from custom_behavior.optical_flow.target_tracker import TargetTracker

# ----------------------------
# Тесты для TargetTracker
# ----------------------------
def test_process_detection_resets_lost():
    # Мокаем детектор
    detector = Mock()
    detector.detect_best_target.return_value = "target"
    
    tracker = TargetTracker(detector, lost_threshold=3)
    
    # Сначала обнаружение
    result = tracker.process("frame")
    assert result == "target"
    assert tracker.lost_frames == 0
    assert tracker.ever_acquired is True

def test_process_detection_none_increases_lost():
    detector = Mock()
    detector.detect_best_target.return_value = None
    
    tracker = TargetTracker(detector, lost_threshold=2)
    
    # Несколько кадров без цели
    for i in range(3):
        result = tracker.process("frame")
        assert result is None
        assert tracker.lost_frames == i + 1
        assert tracker.ever_acquired is False  # пока не было обнаружения

def test_is_lost_behavior():
    detector = Mock()
    tracker = TargetTracker(detector, lost_threshold=2)
    
    # Без обнаружений is_lost всегда False
    for _ in range(5):
        tracker.process("frame")
        assert tracker.is_lost() is False

    # Сначала цель обнаружена
    detector.detect_best_target.return_value = "target"
    tracker.process("frame")
    assert tracker.is_lost() is False

    # Теперь пропущено несколько кадров
    detector.detect_best_target.return_value = None
    tracker.process("frame")  # lost_frames = 1
    assert tracker.is_lost() is False
    tracker.process("frame")  # lost_frames = 2
    assert tracker.is_lost() is False
    tracker.process("frame")  # lost_frames = 3 > threshold
    assert tracker.is_lost() is True

def test_lost_frames_reset_on_new_detection():
    detector = Mock()
    tracker = TargetTracker(detector, lost_threshold=2)
    
    # Обнаружение
    detector.detect_best_target.return_value = "target"
    tracker.process("frame")
    
    # Пропущенные кадры
    detector.detect_best_target.return_value = None
    tracker.process("frame")
    tracker.process("frame")
    assert tracker.lost_frames == 2
    
    # Снова обнаружение → сброс lost_frames
    detector.detect_best_target.return_value = "target"
    tracker.process("frame")
    assert tracker.lost_frames == 0
