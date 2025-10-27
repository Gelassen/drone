### Flying Drone DIY
This project is about process of designing, assembling and programming custom flying drone. The work is mostly done on physics, mechanics and electronics. However, the project has reached the stage when it starts demand custom programming logic and that's a reason behind creating this repo. 

### Drone Versions Summary

| Version | Brief Purpose | Business Intent / Technical Task / Tech Components |
|---------|---------------|---------------------------------------------------|
| **V1** | **Minimal RC drone**<br><small>Assembly and verification of base platform</small> | - **Business intent:** Validate basic flight mechanics and build experience before extending system complexity.<br>- **Technical task:** Assemble 4-motor quadcopter, configure stabilization, test RC link and flight modes (Stabilize / AltHold).<br>- **Tech components:** 450–500 mm frame, 4 motors + 4 ESCs, Cube Orange, basic power module, no FPV or telemetry. |
| **V2** | **Video (FPV) and telemetry**<br><small>Adding live video feedback for operator</small> | - **Business intent:** Enable real-time situational awareness for manual missions and visual monitoring.<br>- **Technical task:** Integrate FPV system with live video and telemetry overlay; test pilot interface and signal stability.<br>- **Tech components:** Analog FPV camera + VTx, OSD module, receiver/display or FPV goggles, telemetry link to ground station. |
| **V3** | **ML integration**<br><small>On-board video processing</small> | - **Business intent:** Move toward semi-autonomous recognition — detect objects and transmit key data to operator.<br>- **Technical task:** Run onboard inference using lightweight neural models (YOLOv5n / YOLOv8n) and forward detections via MAVLink.<br>- **Tech components:** SBC (Jetson Nano / Orange Pi) + CSI / USB camera, local inference pipeline, dual video output for analysis and monitoring. |
| **V4** | **Autonomy and analysis**<br><small>Navigation, SLAM, and target filtering</small> | - **Business intent:** Build semi-autonomous navigation and perception; reduce operator load in mapping or monitoring missions.<br>- **Technical task:** Implement obstacle avoidance and SLAM; process and filter detected targets for smart guidance.<br>- **Tech components:** Depth sensors / LiDAR, SLAM stack (RTAB-Map / PX4 avoidance), web or QGroundControl interface, analytics module. |


Related publications:

<a href="https://gelassen.github.io/blog/2025/10/18/flying-drone-diy-part-III.html">Flying Drone DIY, Part III: Assembling and Tuning</a>

<a href="https://gelassen.github.io/blog/2025/05/17/flying-drone-diy-part-II.html">Flying drone DIY, part II: configuration for the 1st version</a>

<a href="https://gelassen.github.io/blog/2023/03/19/case-study-flying-drone-diy.html">Case study: Flying drone DIY</a>
