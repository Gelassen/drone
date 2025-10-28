import math
import time
from pymavlink import mavutil
import numpy as np

class OpticalFlowProcessor:
    """Обработчик Optical Flow данных через pymavlink, безопасный к разным версиям MAVLink"""
    
    def __init__(self, connection_string, flow_scale=0.001):
        self.master = mavutil.mavlink_connection(connection_string)
        self.flow_scale = flow_scale  # масштабный коэффициент
        self.position = [0.0, 0.0]   # x, y в метрах
        self.last_time = None
    
    def start_receiving(self):
        """Основной цикл приема данных"""
        print("Запуск приема Optical Flow...")
        while True:
            msg = self.master.recv_match(
                type=['OPTICAL_FLOW_RAD', 'OPTICAL_FLOW'],
                blocking=True,
                timeout=1.0
            )
            if msg:
                self.process_optical_flow(msg)
    
    def process_optical_flow(self, msg):
        """Обработка optical flow сообщений, безопасная к разным версиям MAVLink"""
        current_time = time.time()
        msg_type = msg.get_type()

        # Common safe extraction
        flow_x = getattr(msg, 'flow_x', 0.0)
        flow_y = getattr(msg, 'flow_y', 0.0)
        flow_comp_m_x = getattr(msg, 'flow_comp_m_x', 0.0)
        flow_comp_m_y = getattr(msg, 'flow_comp_m_y', 0.0)
        ground_distance = getattr(msg, 'ground_distance', getattr(msg, 'distance', 0.0))
        quality = getattr(msg, 'quality', 0)
        time_usec = getattr(msg, 'time_usec', 0)

        flow_data = {
            'time_usec': time_usec,
            'flow_x': flow_x,
            'flow_y': flow_y,
            'flow_comp_m_x': flow_comp_m_x,
            'flow_comp_m_y': flow_comp_m_y,
            'ground_distance': ground_distance,
            'quality': quality
        }

        self.calculate_velocity(flow_data, current_time)
        self.update_position(flow_data, current_time)
    
    def calculate_velocity(self, flow_data, current_time):
        """Расчет скорости на основе optical flow"""
        if self.last_time is None:
            self.last_time = current_time
            return
        
        dt = current_time - self.last_time
        if dt <= 0:
            return

        # Конвертация flow в скорость (м/с)
        velocity_x = (flow_data['flow_x'] * flow_data['ground_distance'] * self.flow_scale) / dt
        velocity_y = (flow_data['flow_y'] * flow_data['ground_distance'] * self.flow_scale) / dt

        print(f"[Velocity] X={velocity_x:.3f} m/s, Y={velocity_y:.3f} m/s | "
              f"Quality={flow_data['quality']}%, Distance={flow_data['ground_distance']:.2f} m")

        self.last_time = current_time

    def update_position(self, flow_data, current_time):
        """Интегрирование скорости для получения позиции (демо, не EKF)"""
        # Простейшая dead reckoning (только для демонстрации)
        pass
        
class OpticalFlowStabilization:
    """Стабилизация на основе Optical Flow через pymavlink"""
    
    def __init__(self, connection_string):
        self.master = mavutil.mavlink_connection(connection_string)
        self.flow_processor = OpticalFlowProcessor(connection_string)
        self.pid_x = PIDController(kp=1.0, ki=0.1, kd=0.05)
        self.pid_y = PIDController(kp=1.0, ki=0.1, kd=0.05)
        
    def start_stabilization(self):
        """Запуск стабилизации по optical flow"""
        print("Запуск стабилизации по Optical Flow...")
        
        try:
            while True:
                msg = self.master.recv_match(type='OPTICAL_FLOW_RAD', blocking=True, timeout=0.1)
                if msg:
                    correction = self.calculate_stabilization_correction(msg)
                    self.apply_correction(correction)
                    
        except KeyboardInterrupt:
            print("Стабилизация остановлена")
    
    def calculate_stabilization_correction(self, flow_msg):
        """Расчет корректирующих воздействий"""
        # Целевая скорость = 0 (удержание позиции)
        target_velocity_x = 0.0
        target_velocity_y = 0.0
        
        # Текущая скорость из optical flow
        current_velocity_x = flow_msg.integrated_x * 0.1  # упрощенная конвертация
        current_velocity_y = flow_msg.integrated_y * 0.1
        
        # PID коррекция
        correction_x = self.pid_x.update(target_velocity_x, current_velocity_x)
        correction_y = self.pid_y.update(target_velocity_y, current_velocity_y)
        
        return (correction_x, correction_y)
    
    def apply_correction(self, correction):
        """Применение корректирующих команд"""
        # Отправка RC команд или прямого управления
        # В реальности нужно учитывать режим полета и безопасность
        print(f"Коррекция: X={correction[0]:.3f}, Y={correction[1]:.3f}")

class PIDController:
    """Простой PID контроллер"""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0
        
    def update(self, target, current):
        error = target - current
        
        # Пропорциональная составляющая
        p = self.kp * error
        
        # Интегральная составляющая
        self.integral += error
        i = self.ki * self.integral
        
        # Дифференциальная составляющая
        d = self.kd * (error - self.last_error)
        self.last_error = error
        
        return p + i + d

# Демонстрация работы
if __name__ == "__main__":
    # Для симуляции
    connection_str = 'udp:127.0.0.1:14551'

    # Для реального дрона
    # connection_str = #'/dev/ttyACM0'  # или 'udp:192.168.1.1:14550'
    
    # Простой прием данных
    processor = OpticalFlowProcessor(connection_str)
    
    # Или стабилизация
    # stabilizer = OpticalFlowStabilization(connection_str)
    
    # Запуск (выберите один)
    processor.start_receiving()
    # stabilizer.start_stabilization()