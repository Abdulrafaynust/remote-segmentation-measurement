import cv2
import numpy as np

K = np.load("camera_matrix.npy")
dist = np.load("distortion.npy")

img = cv2.imread("calibration/img_001.png")
h, w = img.shape[:2]

newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
img = cv2.undistort(img, K, dist, None, newK)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (7,11), None)

if ret:
    p1 = corners[0][0]
    p2 = corners[1][0]

    pixel_dist = np.linalg.norm(p1 - p2)

    square_size_mm = 25

    pixel_to_mm = square_size_mm / pixel_dist

    print("Pixel distance:", pixel_dist)
    print("Pixel to mm:", pixel_to_mm)