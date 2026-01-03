from collections import deque

from collections import deque

class SignalBuffer:

    def __init__(
        self,
        max_frames=60,
        expected_frame_dt=0.033,
        max_gap_frames=3
    ):
        self.max_frames = max_frames
        self.expected_frame_dt = expected_frame_dt
        self.max_gap = max_gap_frames * expected_frame_dt

        self.data = {}      # name -> deque[(value, ts)]
        self.last_ts = {}   # name -> ts

    def reset(self, name):
        self.data[name] = deque()
        self.last_ts[name] = None

    def update(self, signal):
        name = signal.name

        if name not in self.data:
            self.data[name] = deque()
            self.last_ts[name] = None

        last_ts = self.last_ts[name]
        if last_ts is not None:
            if signal.ts - last_ts > self.max_gap:
                # временной разрыв → история невалидна
                self.reset(name)

        self.data[name].append((signal.value, signal.ts))

        if len(self.data[name]) > self.max_frames:
            self.data[name].popleft()

        self.last_ts[name] = signal.ts

    def values(self, name):
        if name not in self.data:
            return []
        return [v for v, _ in self.data[name]]

    def timestamps(self, name):
        if name not in self.data:
            return []
        return [ts for _, ts in self.data[name]]

    def size(self, name):
        return len(self.data.get(name, []))

