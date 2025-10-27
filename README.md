### Flying Drone DIY
This project is about process of designing, assembling and programming custom flying drone. The work is mostly done on physics, mechanics and electronics. However, the project has reached the stage when it starts demand custom programming logic and that's a reason behind creating this repo. 

                <table aria-label="Drone versions summary">
                    <thead>
                    <tr>
                        <th style="width:110px;">Version</th>
                        <th style="width:260px;">Brief purpose</th>
                        <th>Business intent / Technical task / Tech components</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr>
                        <td><span class="badge">V1</span></td>
                        <td><strong>Minimal RC drone</strong><br/><small>Assembly and verification of base platform</small></td>
                        <td>
                        <ul class="features">
                            <li><strong>Business intent:</strong> Validate basic flight mechanics and build experience before extending system complexity.</li>
                            <li><strong>Technical task:</strong> Assemble 4-motor quadcopter, configure stabilization, test RC link and flight modes (Stabilize / AltHold).</li>
                            <li><strong>Tech components:</strong> 450–500 mm frame, 4 motors + 4 ESCs, Cube Orange, basic power module, no FPV or telemetry.</li>
                        </ul>
                        </td>
                    </tr>

                    <tr>
                        <td><span class="badge">V2</span></td>
                        <td><strong>Video (FPV) and telemetry</strong><br/><small>Adding live video feedback for operator</small></td>
                        <td>
                        <ul class="features">
                            <li><strong>Business intent:</strong> Enable real-time situational awareness for manual missions and visual monitoring.</li>
                            <li><strong>Technical task:</strong> Integrate FPV system with live video and telemetry overlay; test pilot interface and signal stability.</li>
                            <li><strong>Tech components:</strong> Analog FPV camera + VTx, OSD module, receiver/display or FPV goggles, telemetry link to ground station.</li>
                        </ul>
                        </td>
                    </tr>

                    <tr>
                        <td><span class="badge">V3</span></td>
                        <td><strong>ML integration</strong><br/><small>On-board video processing</small></td>
                        <td>
                        <ul class="features">
                            <li><strong>Business intent:</strong> Move toward semi-autonomous recognition — detect objects and transmit key data to operator.</li>
                            <li><strong>Technical task:</strong> Run onboard inference using lightweight neural models (YOLOv5n / YOLOv8n) and forward detections via MAVLink.</li>
                            <li><strong>Tech components:</strong> SBC (Jetson Nano / Orange Pi) + CSI / USB camera, local inference pipeline, dual video output for analysis and monitoring.</li>
                        </ul>
                        </td>
                    </tr>

                    <tr>
                        <td><span class="badge">V4</span></td>
                        <td><strong>Autonomy and analysis</strong><br/><small>Navigation, SLAM, and target filtering</small></td>
                        <td>
                        <ul class="features">
                            <li><strong>Business intent:</strong> Build semi-autonomous navigation and perception; reduce operator load in mapping or monitoring missions.</li>
                            <li><strong>Technical task:</strong> Implement obstacle avoidance and SLAM; process and filter detected targets for smart guidance.</li>
                            <li><strong>Tech components:</strong> Depth sensors / LiDAR, SLAM stack (RTAB-Map / PX4 avoidance), web or QGroundControl interface, analytics module.</li>
                        </ul>
                        </td>
                    </tr>
                    </tbody>
                </table>

Related publications:

<a href="https://gelassen.github.io/blog/2025/10/18/flying-drone-diy-part-III.html">Flying Drone DIY, Part III: Assembling and Tuning</a>
<a href="https://gelassen.github.io/blog/2025/05/17/flying-drone-diy-part-II.html">Flying drone DIY, part II: configuration for the 1st version</a>
<a href="https://gelassen.github.io/blog/2023/03/19/case-study-flying-drone-diy.html">Case study: Flying drone DIY</a>
