#!/bin/bash
# ------------------------------------------------------------------
# Скрипт запуска ArduPilot SITL + Gazebo (iris model)
# ------------------------------------------------------------------

# -------------------- ПЕРЕМЕННЫЕ --------------------
# Путь к папке ArduPilot
ARDUPILOT_PATH="$HOME/Workspace/Personal/drone/ardupilot"

# Путь к Gazebo world или sdf файлу
GAZEBO_MODEL_PATH="$HOME/Workspace/Personal/drone/ardupilot_gazebo/worlds/iris_runway.sdf"

# Виртуальная среда
VENV_PATH="$HOME/Workspace/Personal/drone/.venv"
REQUIREMENTS_FILE="$HOME/Workspace/Personal/drone/requirements.txt"

# UDP порты связи SITL <-> Gazebo
GAZEBO_OUT_PORT=9002  # Gazebo → SITL
GAZEBO_IN_PORT=9003   # SITL → Gazebo

# -------------------- ПОДГОТОВКА --------------------
echo "Активация виртуальной среды..."
source "$VENV_PATH/bin/activate"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Проверка зависимостей..."
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Файл requirements.txt не найден: $REQUIREMENTS_FILE"
fi

export GZ_SIM_RESOURCE_PATH="$HOME/Workspace/Personal/drone/ardupilot_gazebo/models"

# -------------------- ЗАПУСК SITL --------------------
echo "Запуск ArduCopter SITL..."
gnome-terminal -- bash -c "
cd $ARDUPILOT_PATH/Tools/autotest &&
source $VENV_PATH/bin/activate &&
python3 sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console 
exec bash
"

sleep 5  # даём SITL время на инициализацию

# -------------------- ЗАПУСК GAZEBO --------------------
echo "Запуск Gazebo..."
gnome-terminal -- bash -c "
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH &&
gz sim -v4 -r  $GAZEBO_MODEL_PATH
exec bash
"

# -------------------- ИНФОРМАЦИЯ --------------------
echo "---------------------------------------------"
echo "✅ SITL ↔ Gazebo связаны через UDP:"
echo "   Gazebo → SITL: 127.0.0.1:$GAZEBO_OUT_PORT"
echo "   SITL → Gazebo: 127.0.0.1:$GAZEBO_IN_PORT"
echo ""
echo "Если дрон не двигается:"
echo "  1. Проверь, что плагины libArduPilotPlugin.so загружены в Gazebo"
echo "  2. Убедись, что в iris.sdf указаны те же порты"
echo "---------------------------------------------"
