# hardware_interface.py
from abc import ABC, abstractmethod
from mavsdk.offboard import VelocityNedYaw

class HardwareInterface(ABC):

    @abstractmethod
    def is_connected(self):
        pass

    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def arm_and_takeoff(self, target_alt_m: float):
        pass

    @abstractmethod
    async def start_offboard(self):
        pass

    @abstractmethod
    async def stop_offboard(self):
        pass

    @abstractmethod
    async def land(self):
        pass

    @abstractmethod
    async def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate=0):
        pass