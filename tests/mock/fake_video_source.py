class FakeVideoSource:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.frame_w = 640
        self.frame_h = 480

    async def open(self):
        pass

    async def read(self):
        if self.idx < len(self.frames):
            f = self.frames[self.idx]
            self.idx += 1
            return f
        return None

    def close(self):
        pass