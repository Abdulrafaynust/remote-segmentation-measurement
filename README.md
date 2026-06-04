# Smart Object Measurement using Camera Calibration and Deep Learning

## Project Overview
This project builds an end-to-end computer vision pipeline that:
- Calibrates a camera using checkerboard images
- Removes lens distortion using intrinsic parameters
- Collects and labels a custom dataset of a TV remote
- Trains a UNet segmentation model
- Extracts object masks from images
- Converts pixel measurements into real-world millimetres

---

## Pipeline Workflow
1. Camera Calibration using OpenCV
2. Image Undistortion
3. Dataset Collection & Labeling
4. Model Training (UNet)
5. Inference (Segmentation)
6. Pixel-to-mm Measurement

---

## How to Run

### Step 1: Camera Calibration
python calibrate_camera.py

### Step 2: Train Model
python train.py

### Step 3: Run Full Pipeline
python pipeline.py

---

## Output
- Segmentation mask of object
- Bounding box around object
- Real-world width & height in millimetres

---

## Tools Used
- Python
- OpenCV
- PyTorch
- NumPy

