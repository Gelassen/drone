from pymavlink import mavutil
import time

# Подключение к SITL
master = mavutil.mavlink_connection('udp:127.0.0.1:14551')
master.wait_heartbeat()
print("Connected to system:", master.target_system, master.target_component)

# Включение потока данных (GPS, высота и т.д.)
master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    2, 1
)

# --- Функции управления ---

def set_mode(mode):
    """Меняем режим полёта через MAV_CMD_DO_SET_MODE"""
    # Режимы ArduCopter (GUIDED = 4, LOITER = 5, RTL = 6, ...)
    # Полный список: https://ardupilot.org/copter/docs/flight-modes.html
    mode_id = {
        'GUIDED': 4,
        'LOITER': 5,
        'RTL': 6
    }.get(mode.upper(), 4)

    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    print(f"Mode set to {mode}")
    time.sleep(2)

def arm_and_takeoff(target_altitude):
    # Установка режима GUIDED
    set_mode('GUIDED')

    # Армирование
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    print("Arming...")
    time.sleep(3)

    # Взлёт
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_altitude
    )
    print(f"Taking off to {target_altitude} meters")
    time.sleep(10)  # Ждём набора высоты


def goto_relative(dx=0, dy=0, dz=0):
    """
    Двигаемся относительно текущей позиции:
    dx, dy в метрах по северу/востоку
    dz в метрах вниз (положительное dz — вниз)
    """
    master.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
        10,  # time_boot_ms
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,  # маска (позиция)
        dx, dy, -dz,  # N, E, D (D вниз, поэтому отрицательное)
        0, 0, 0,      # скорости (vx,vy,vz)
        0, 0, 0,      # ускорение
        0, 0          # yaw, yaw_rate
    ))
    print(f"Moving to relative position N:{dx} E:{dy} D:{dz}")
    time.sleep(8)

def return_to_launch():
    master.set_mode_rtl()
    print("Returning to launch")
    time.sleep(10)

# --- Полётная последовательность ---
arm_and_takeoff(100)   # Взлёт до 100 м
goto_relative(dx=80)   # Прямо на 80 м
goto_relative(dz=50)   # Снижение до 50 м
goto_relative(dx=-80, dz=0)  # Возврат к точке старта
return_to_launch()     # RTL на всякий случай

print("Flight sequence completed")
