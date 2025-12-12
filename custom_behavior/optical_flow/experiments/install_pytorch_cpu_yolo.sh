#!/bin/bash

echo "==== Removing old PyTorch ===="
pip uninstall -y torch torchvision torchaudio

echo "==== Installing CPU PyTorch ===="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "==== Installing Ultralytics (YOLOv8) ===="
pip install ultralytics

echo "==== Checking PyTorch ===="
python3 - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

echo "==== Checking YOLOv8 ===="
python3 - << 'EOF'
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
print("YOLO loaded successfully!")
EOF

echo "==== All done! ===="
echo "PyTorch работает в CPU-режиме, YOLOv8 установлена и готова к запуску."
