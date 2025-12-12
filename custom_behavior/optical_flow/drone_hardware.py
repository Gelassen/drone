import asyncio

from mavsdk import System
from mavsdk.offboard import (
    OffboardError, 
    VelocityNedYaw
)

from hardware_interface import HardwareInterface


class DroneHardware(HardwareInterface):

    def __init__(
            self,
            connection_url="udpin://127.0.0.1:14550",
            disable_mav = False,
            connected = False
        ):
        print("DroneHardware::__init__ call")
        self.connection_url = connection_url
        self.disable_mav = disable_mav
        self._connected = connected
        self.drone = System() if not self.disable_mav else None

    def is_connected(self):
        return self.is_connected
    
    async def connect(self):
        if self.disable_mav:
            print("MAV disabled, skipping connect")
            return
        
        print("Connecting to drone...")
        try:
            await self.drone.connect(system_address=self.connection_url)
        except Exception as e:
            # connect may raise if bad URL; continue but mark disconnected
            print(f"Warning: drone.connect() exception: {e}")
            self._connected = False
            return

        try:
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    print("Drone discovered!")
                    self._connected = True
                    break
        except Exception as e:
            print(f"Warning: connection_state() exception: {e}")
            self._connected = False
    
    async def arm_and_takeoff(self, target_alt_m: float = 1.5):
        # self.log.append(f"takeoff {target_alt_m}")
        if self.disable_mav or not self._connected:
            print("Skipping arm/takeoff (disabled or not connected)")
            return
        
        print("Arming...")
        try:
            await self.drone.action.arm()
            print(f"Taking off to {target_alt_m} m...")
            await self.drone.action.takeoff()
            async for pos in self.drone.telemetry.position():
                if pos.relative_altitude_m >= target_alt_m * 0.95:
                    print("Reached target altitude")
                    break
                await asyncio.sleep(0.2)
        except Exception as e:
            print(f"Warning: arm/takeoff failed: {e}")
            # don't raise — continue in safe mode
            self._connected = False
    
    async def start_offboard(self):
        print("[start] DroneHardware::start_offboard")
        if self.disable_mav or not self._connected:
            print("DroneHardware::start_offboard -- forbidden or not connected. Exit.")
            return
        try:
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
            await self.drone.offboard.start()
            print("Offboard started")
        except OffboardError as e:
            print(f"Failed to start Offboard: {e._result.result}")
            raise
        except Exception as e:
            print(f"Warning: start_offboard failed: {e}")
            self._connected = False

    async def stop_offboard(self):
        if self.disable_mav or not self._connected:
            return
        try:
            await self.drone.offboard.stop()
            print("Offboard stopped")
        except Exception as e:
            print(f"Warning: stop_offboard failed: {e}")

    async def land(self):
        try:
            await self.drone.action.land()
        except Exception:
            pass

    async def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate=0):
        """
        Посылает скорости по NED.
        Параметры:
            vx: скорость по North (м/с)
            vy: скорость по East (м/с)
            vz: скорость по Down (м/с, положительное вниз)
            yaw_rate: угловая скорость вокруг вертикали (deg/s)
        """
        result = True
        if self.disable_mav:
            print("drone_harwdare::send_velocity")
            return False
        
        if not self._connected:
            await self.connect()
        try:
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, vz, yaw_rate))
        except Exception as e:
            print("[MAVSDKHardware] send_velocity failed:", e)
            result = False
            
        return True