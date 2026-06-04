import cv2
import numpy as np
import torch

from model import UNet

# ----------------------------
# LOAD CALIBRATION
# ----------------------------
K = np.load("camera_matrix.npy")
dist = np.load("distortion.npy")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# LOAD MODEL
# ----------------------------
model = UNet().to(device)
model.load_state_dict(torch.load("unet_remote.pth", map_location=device))
model.eval()

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------
def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0).to(device)

# ----------------------------
# PIPELINE
# ----------------------------
def run_pipeline(image_path, pixel_to_mm=0.6221473):

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found")
        return

    h, w = img.shape[:2]

    # ---- UNDISTORT ----
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    img = cv2.undistort(img, K, dist, None, newK)

    # ---- SEGMENTATION ----
    inp = preprocess(img)

    with torch.no_grad():
        pred = model(inp)[0][0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # ---- CONTOUR ----
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("❌ No object detected")
        return

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # ---- MEASUREMENT ----
    width_mm = w * pixel_to_mm
    height_mm = h * pixel_to_mm

    # ---- OUTPUT ----
    print("\n📏 FINAL RESULT")
    print("------------------")
    print("Width (mm):", round(width_mm, 2))
    print("Height (mm):", round(height_mm, 2))

    # ---- VISUALIZATION ----
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Result", vis)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)


# ----------------------------
# RUN TEST
# ----------------------------
run_pipeline("test_final.jpeg", pixel_to_mm=0.12)