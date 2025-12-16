import sys
import types

# мок для pupil_apriltags
# class DummyDetector:
#     def __init__(self, *args, **kwargs):
#         pass
#     def detect(self, gray_image, camera_params=None, tag_size=None):
#         return []

# # подменяем в sys.modules
# sys.modules['pupil_apriltags'] = types.SimpleNamespace(Detector=lambda *a, **k: DummyDetector())
