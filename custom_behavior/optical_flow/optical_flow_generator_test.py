from pymavlink import mavutil

master = mavutil.mavlink_connection('udp:127.0.0.1:14551')
while True:
    msg = master.recv_match(type=['OPTICAL_FLOW', 'OPTICAL_FLOW_RAD'], blocking=True, timeout=1)
    if msg:
        print(msg)
