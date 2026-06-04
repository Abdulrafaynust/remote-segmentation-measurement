import cv2
import numpy as np
import os

K = np.load("camera_matrix.npy")
dist = np.load("distortion.npy")

folder = "dataset/train/images"

files = os.listdir(folder)

if len(files) == 0:
    print("❌ No images found in folder")
    exit()

img_path = os.path.join(folder, files[0])  # take FIRST available image

img = cv2.imread(img_path)

if img is None:
    print(" Failed to load:", img_path)
    exit()

h, w = img.shape[:2]

newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, K, dist, None, newK)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)