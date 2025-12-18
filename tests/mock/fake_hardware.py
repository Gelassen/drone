from custom_behavior.optical_flow.hardware_interface import HardwareInterface

class FakeHardware(HardwareInterface):
    def __init__(self, can_arm=True):
        self._connected = False
        self._armed = False
        self._offboard = False
        self._can_arm = can_arm
        self._land_called = False
        self.calls = []

    def is_connected(self):
        return self._connected

    def is_land_called(self):
        return self._land_called
    
    def is_offboard(self):
        return self._offboard
    
    def is_armed(self):
        return self._armed

    async def connect(self):
        self.calls.append("connect")
        self._connected = True

    async def can_arm(self) -> bool:
        self.calls.append("can_arm")
        return self._can_arm

    async def can_arm_with_backoff(self) -> bool:
        self.calls.append("can_arm_with_backoff")
        return self._can_arm

    async def arm_and_takeoff(self, target_alt_m: float):
        self.calls.append(("arm_and_takeoff", target_alt_m))
        if not self._can_arm:
            raise RuntimeError("Cannot arm")
        self._armed = True

    async def start_offboard(self):
        self.calls.append("start_offboard")
        self._offboard = True

    async def stop_offboard(self):
        self.calls.append("stop_offboard")
        self._offboard = False

    async def land(self):
        self.calls.append("land")
        self._armed = False
        self._land_called=True

    async def send_velocity(self, vx, vy, vz, yaw_rate=0):
        self.calls.append(("send_velocity", vx, vy, vz, yaw_rate))
