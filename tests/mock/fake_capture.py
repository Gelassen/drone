# ----------------------------
# Фикстура мок-VideoCapture
# ----------------------------
class FakeCapture:
    def __init__(self, ret=True, frame=None, w=640, h=480):
        self.ret = ret
        self.frame = frame if frame is not None else "frame"
        self.w = w
        self.h = h
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop_id):
        if prop_id == 3:  # cv2.CAP_PROP_FRAME_WIDTH
            return self.w
        if prop_id == 4:  # cv2.CAP_PROP_FRAME_HEIGHT
            return self.h
        return 0

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.released = True