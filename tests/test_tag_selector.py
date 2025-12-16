import pytest
import numpy as np
from custom_behavior.optical_flow.tag_selector import TagSelector
from custom_behavior.optical_flow.tag_geometry import TagGeometry

# ----------------------------
# Мок-объект для тега
# ----------------------------
class MockTag:
    def __init__(self, corners):
        self.corners = corners

# ----------------------------
# Тесты для TagSelector
# ----------------------------
def test_select_best_empty():
    selector = TagSelector()
    assert selector.select_best([]) is None

def test_select_best_single_tag():
    selector = TagSelector()
    tag = MockTag(corners=np.array([[0,0],[0,1],[1,1],[1,0]], dtype=float))
    assert selector.select_best([tag]) == tag

def test_select_best_multiple_tags():
    selector = TagSelector()
    tag1 = MockTag(corners=np.array([[0,0],[0,1],[1,1],[1,0]], dtype=float))  # px_size = 1
    tag2 = MockTag(corners=np.array([[0,0],[0,2],[2,2],[2,0]], dtype=float))  # px_size = 2
    tag3 = MockTag(corners=np.array([[0,0],[0,1],[2,1],[2,0]], dtype=float))  # px_size = 1.5
    best = selector.select_best([tag1, tag2, tag3])
    assert best == tag2

def test_select_best_handles_invalid_corners(monkeypatch):
    selector = TagSelector()
    # Тег с пустыми corners → px_size_from_corners вернёт ошибку, можно замокать
    tag1 = MockTag(corners=np.zeros((0, 2)))
    tag2 = MockTag(corners=np.array([[0,0],[0,1],[1,1],[1,0]], dtype=float))
    
    # Принудительно вернуть 0 для некорректных corners
    monkeypatch.setattr(TagGeometry, "px_size_from_corners", lambda x: 0 if len(x) == 0 else 1)
    
    best = selector.select_best([tag1, tag2])
    assert best == tag2
