import cv2
import asyncio

class AsyncVideoSource:
    def __init__(self, src, executor):
        self.src = src
        self.executor = executor
        self.cap = None
        self.frame_w = None
        self.frame_h = None

    def _open_sync(self):
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.src}")
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    async def open(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._open_sync)

    def _read_sync(self):
        return self.cap.read()

    async def read(self):
        loop = asyncio.get_running_loop()
        ret, frame = await loop.run_in_executor(self.executor, self._read_sync)
        return frame if ret else None

    def close(self):
        if self.cap:
            self.cap.release()
