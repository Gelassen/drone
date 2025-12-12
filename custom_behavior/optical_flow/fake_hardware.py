# fake_hardware.py
from .hardware_interface import HardwareInterface

class FakeHardware(HardwareInterface):
    def __init__(self):
        self.log = []

    async def connect(self):
        self.log.append("connect")

    async def arm_and_takeoff(self, target_alt_m: float):
        self.log.append(f"takeoff {target_alt_m}")

    async def start_offboard(self):
        self.log.append("offboard start")

    async def stop_offboard(self):
        self.log.append("offboard stop")

    async def land(self):
        self.log.append("land")

    async def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate=0):
        self.log.append((vx, vy, vz, yaw_rate))
