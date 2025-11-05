import asyncio
import math
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw

# -------------------- ФУНКЦИИ --------------------
async def connect_drone():
    drone = System()
    await drone.connect(system_address="udpin://127.0.0.1:14552")

    print("Connecting to drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone discovered!")
            break
    return drone

async def arm_and_takeoff(drone):
    default_alt = 1.7 # by default 2 meters is a target alt on takeoff (1.7 for gazebo iris_runaway)
    print("Arming drone...")
    await drone.action.arm()

    print(f"Taking off to {default_alt} meters...")
    await drone.action.takeoff()

    # Контроль фактической высоты
    async for pos in drone.telemetry.position():
        alt = pos.relative_altitude_m
        print(f" Altitude: {alt:.1f} m")
        if alt >= default_alt * 0.95:
            print("Reached target altitude")
            break
        await asyncio.sleep(0.5)

async def start_offboard(drone):
    # Обнуляем скорости перед Offboard
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    try:
        await drone.offboard.start()
        print("Offboard started")
    except OffboardError as e:
        print(f"Failed to start Offboard: {e._result.result}")
        return False
    return True

async def reach_altitude(drone, target_alt):
    async for position in drone.telemetry.position():
        alt = position.relative_altitude_m
        print(f"Altitude: {alt:.1f} m")
        if alt >= target_alt:
            print("Reached target altitude")
            break
        # Отправляем команду вверх каждый цикл
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -target_alt, 0.0))
        await asyncio.sleep(0.5)

async def goto_position(drone, north_m, east_m, down_m, yaw_deg=0):
    print(f"Moving to N:{north_m} E:{east_m} D:{down_m}")
    try:
        await drone.offboard.set_position_ned(PositionNedYaw(north_m, east_m, down_m, yaw_deg))
    except OffboardError as e:
        print(f"Offboard error: {e._result.result}")

    # Контроль позиции
    async for pos in drone.telemetry.position_velocity_ned():
        dn = north_m - pos.position.north_m
        de = east_m - pos.position.east_m
        dd = down_m - pos.position.down_m
        dist = math.sqrt(dn**2 + de**2 + dd**2)
        print(f" Distance to target: {dist:.1f} m")
        if dist <= 1.0:
            print("Reached target position")
            break
        await asyncio.sleep(0.5)

# -------------------- MAIN --------------------
async def main():
    drone = await connect_drone()

    await arm_and_takeoff(drone)

    # Включаем Offboard
    offboard_ready = await start_offboard(drone)
    if not offboard_ready:
        print("Cannot start Offboard. Abort mission.")
        return

    await reach_altitude(drone, 50)

    # Прямой полёт на 80 м на север (NED координаты)
    await goto_position(drone, north_m=80, east_m=0, down_m=-80)

    # Снижение до 50 м
    await goto_position(drone, north_m=80, east_m=0, down_m=-50) 

    # Возврат на точку старта
    await goto_position(drone, north_m=0, east_m=0, down_m=-10)

    # Выключаем Offboard перед посадкой
    try:
        await drone.offboard.stop()
        print("Offboard stopped")
    except OffboardError as e:
        print(f"Failed to stop Offboard: {e._result.result}")

    # Посадка
    print("Landing...")
    await drone.action.land()
    
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            break

    async for is_armed in drone.telemetry.armed():
        if not is_armed:
            break

    print("Landed and disarmed. Disconnecting.")
    await drone.disconnect()
    print("Mission complete")

if __name__ == "__main__":
    asyncio.run(main())
