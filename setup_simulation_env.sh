#!/bin/bash
# ------------------------------------------------------------------
# Скрипт запуска ArduPilot SITL + QGroundControl на Linux
# ------------------------------------------------------------------

# -------------------- ПЕРЕМЕННЫЕ --------------------
# Путь к папке ArduPilot (где находится sim_vehicle.py)
ARDUPILOT_PATH="$HOME/Workspace/Personal/drone/ardupilot"

# Путь к QGroundControl AppImage
QGC_PATH="$HOME/Workspace/Personal/drone/QGroundControl-x86_64.AppImage"

# Путь к виртуальной среде и requirements
VENV_PATH="$HOME/Workspace/Personal/drone/.venv"
REQUIREMENTS_FILE="$HOME/Workspace/Personal/drone/requirements.txt"

# MAVLink порты
UDP_PORT="14551" # порт для MAVLink, изменён для избежания конфликта
TCP_PORT="5760"

# -------------------- ВИРТУАЛЬНАЯ СРЕДА --------------------
echo "Создание виртуальной среды (venv)..."
python3 -m venv "$VENV_PATH"

echo "Активация виртуальной среды..."
source "$VENV_PATH/bin/activate"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Установка зависимостей из requirements.txt..."
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Файл requirements.txt не найден: $REQUIREMENTS_FILE"
fi

# -------------------- ЗАПУСК --------------------
echo "Запуск SITL ArduCopter..."
cd "$ARDUPILOT_PATH/Tools/autotest"

echo "Запуск SITL с консолью MAVProxy..."
gnome-terminal -- bash -c "source $VENV_PATH/bin/activate && cd $ARDUPILOT_PATH/Tools/autotest && ./sim_vehicle.py -v ArduCopter --console --out=udp:127.0.0.1:$UDP_PORT; exec bash"

sleep 3  # даём SITL немного времени на инициализацию

echo "Запуск QGroundControl..."
chmod +x "$QGC_PATH"
"$QGC_PATH" &

QGC_PID=$!

# -------------------- ИНФОРМАЦИЯ --------------------
echo "SITL PID: $SITL_PID"
echo "QGroundControl PID: $QGC_PID"
echo "UDP-порт для MAVLink: $UDP_PORT"
echo "TCP-порт для альтернативного подключения: $TCP_PORT"

# -------------------- ОЖИДАНИЕ --------------------
echo "Ожидание завершения процессов..."
wait $QGC_PID

# -------------------- НАПОМИНАНИЕ О ДЕАКТИВАЦИЯ ВИРТУАЛЬНОЙ СРЕДЫ --------------------
echo "Не забудьте деактивировать виртуальную среду."
