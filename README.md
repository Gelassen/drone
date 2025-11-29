### Flying Drone DIY
This project is about process of designing, assembling and programming custom flying drone. The work is mostly done on physics, mechanics and electronics. However, the project has reached the stage when it starts demand custom programming logic and that's a reason behind creating this repo. 

### Drone Versions Summary

| Version | Brief Purpose | Business Intent / Technical Task / Tech Components |
|---------|---------------|---------------------------------------------------|
| **V1** | **Minimal RC drone**<br><small>Assembly and verification of base platform</small> | - **Business intent:** Validate basic flight mechanics and build experience before extending system complexity.<br>- **Technical task:** Assemble 4-motor quadcopter, configure stabilization, test RC link and flight modes (Stabilize / AltHold).<br>- **Tech components:** 450–500 mm frame, 4 motors + 4 ESCs, Cube Orange, basic power module, no FPV or telemetry. |
| **V2** | **Video (FPV) and telemetry**<br><small>Adding live video feedback for operator</small> | - **Business intent:** Enable real-time situational awareness for manual missions and visual monitoring.<br>- **Technical task:** Integrate FPV system with live video and telemetry overlay; test pilot interface and signal stability.<br>- **Tech components:** Analog FPV camera + VTx, OSD module, receiver/display or FPV goggles, telemetry link to ground station. |
| **V3** | **ML integration**<br><small>On-board video processing</small> | - **Business intent:** Move toward semi-autonomous recognition — detect objects and transmit key data to operator.<br>- **Technical task:** Run onboard inference using lightweight neural models (YOLOv5n / YOLOv8n) and forward detections via MAVLink.<br>- **Tech components:** SBC (Jetson Nano / Orange Pi) + CSI / USB camera, local inference pipeline, dual video output for analysis and monitoring. |
| **V4** | **Autonomy and analysis**<br><small>Navigation, SLAM, and target filtering</small> | - **Business intent:** Build semi-autonomous navigation and perception; reduce operator load in mapping or monitoring missions.<br>- **Technical task:** Implement obstacle avoidance and SLAM; process and filter detected targets for smart guidance.<br>- **Tech components:** Depth sensors / LiDAR, SLAM stack (RTAB-Map / PX4 avoidance), web or QGroundControl interface, analytics module. |

### Deployment
Start simulation environment first and run flights second
```
$ sudo apt update
$ sudo apt install git python3 python3-pip python3-dev \
    python3-future python3-lxml python3-pyproj python3-matplotlib \
    python3-opencv python3-numpy python3-pip \
    build-essential pkg-config \
    libxml2-dev libxslt-dev \
    libudev-dev \
    genromfs

$ cd ~
$ git clone https://github.com/ArduPilot/ardupilot.git
$ cd ardupilot
$ git submodule update --init --recursive

$ Tools/environment_install/install-prereqs-ubuntu.sh -y
$ . ~/.profile

$ wget https://github.com/mavlink/qgroundcontrol/releases/download/latest/QGroundControl.AppImage
$ chmod +x QGroundControl.AppImage
```

To run simulation with gazebo (more complete physical engine) run this:
```
$ sudo apt-get update
$ sudo apt-get install curl lsb-release gnupg

$ sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-prerelease $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-prerelease.list > /dev/null
$ sudo apt-get update
$ sudo apt-get install gz-jetty

$ cd ~
$ git clone https://github.com/ArduPilot/ardupilot_gazebo.git
$ cd ardupilot_gazebo
$ mkdir build && cd build
$ export GZ_VERSION="jetty"
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j$(nproc)
$ sudo make install


$ chmod +x setup_simulations_env.sh && ./setup_simulations_env.sh
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ cd "custom behavior"
$ python <filename>.py

$ echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/gz_ws/src/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}' >> ~/.bashrc
$ echo 'export GZ_SIM_RESOURCE_PATH=$HOME/gz_ws/src/ardupilot_gazebo/models:$HOME/gz_ws/src/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc

$ chmod +x run_sitl_gazebo.sh
$ ./run_sitl_gazebo.sh
```

Related publications:

<a href="https://gelassen.github.io/blog/2025/10/18/flying-drone-diy-part-III.html">Flying Drone DIY, Part III: Assembling and Tuning</a>

<a href="https://gelassen.github.io/blog/2025/05/17/flying-drone-diy-part-II.html">Flying drone DIY, part II: configuration for the 1st version</a>

<a href="https://gelassen.github.io/blog/2023/03/19/case-study-flying-drone-diy.html">Case study: Flying drone DIY</a>
