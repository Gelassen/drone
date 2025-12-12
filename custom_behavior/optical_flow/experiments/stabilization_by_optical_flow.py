import asyncio
from mavsdk import System

async def run():
    drone = System()
    await drone.connect(system_address="udpin://127.0.0.1:14551")

    print("Ожидание подключения...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Подключено к дрону!")
            break

    # Подписка на сообщения Optical Flow (классическое)
    async for flow in drone.telemetry.optical_flow_pix():  # OPTICAL_FLOW
        print(f"Flow X: {flow.flow_x:.2f} px, Flow Y: {flow.flow_y:.2f} px")
        print(f"Flow Compensated: {flow.flow_comp_m_x:.4f} m, {flow.flow_comp_m_y:.4f} m")
        print(f"Ground Distance: {flow.ground_distance_m:.2f} m, Quality: {flow.quality}")
        print(f"Gyro X/Y/Z: {flow.gyro_x:.4f}, {flow.gyro_y:.4f}, {flow.gyro_z:.4f}")

asyncio.run(run())
