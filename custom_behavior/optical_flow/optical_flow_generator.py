from pymavlink import mavutil
import time

master = mavutil.mavlink_connection('udpout:127.0.0.1:14551')

try:
    while True:
        master.mav.optical_flow_send(
            time_usec=int(time.time() * 1e6),
            sensor_id=0,
            flow_x=10, flow_y=-8,
            flow_comp_m_x=0.01, flow_comp_m_y=-0.02,
            quality=250,
            ground_distance=1.2
        )
        print("Sent a new packet of optical flow data")
        time.sleep(0.1)
except Exception as ex:
    print("Got an exeption", ex)

