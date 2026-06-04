import cv2
import numpy as np
import os

# -----------------------------
# SETUP PATHS
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

image_folder = os.path.join(base_dir, "dataset/images")
mask_folder = os.path.join(base_dir, "dataset/masks")

os.makedirs(mask_folder, exist_ok=True)

# -----------------------------
# GET ALL IMAGES
# -----------------------------
image_files = sorted(os.listdir(image_folder))

print(f"Found {len(image_files)} images")

# -----------------------------
# PROCESS EACH IMAGE
# -----------------------------
for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print("Skipping:", img_name)
        continue

    # -------------------------
    # 1. Convert to grayscale
    # -------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # 2. Detect black object (remote)
    # -------------------------
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # -------------------------
    # 3. Clean noise
    # -------------------------
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # -------------------------
    # 4. Find contours
    # -------------------------
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # -------------------------
    # 5. Create empty mask
    # -------------------------
    mask = np.zeros(gray.shape, dtype=np.uint8)

    if contours:
        # largest contour = remote
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    # -------------------------
    # 6. Save mask
    # -------------------------
    mask_name = img_name.rsplit(".", 1)[0] + ".png"
    save_path = os.path.join(mask_folder, mask_name)

    cv2.imwrite(save_path, mask)

    print("Processed:", img_name)

print("DONE!  All masks created")