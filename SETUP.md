# Setup Instructions

## Environment Setup

Install required dependencies:

pip install -r requirements.txt

---

## Required Libraries
- opencv-python
- torch
- torchvision
- numpy

---

## Project Execution Order

1. Camera Calibration
   python calibrate_camera.py

2. Model Training
   python train.py

3. Inference + Measurement
   python pipeline.py

---

## Notes
- Ensure dataset paths are correct
- Ensure calibration images are properly captured
- Always run calibration before measurement
