import cv2
import numpy as np
import os

# -------------------------
# SETTINGS (CHANGE IF NEEDED)
# -------------------------
CHECKERBOARD = (7, 11)  # inner corners (adjust if your board differs)
square_size = 25  # mm (only needed for documentation, not calibration math)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []

# prepare object points (0,0,0), (1,0,0) ...
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

folder = "calibration"
images = os.listdir(folder)

for file in images:
    path = os.path.join(folder, file)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# -------------------------
# CALIBRATION
# -------------------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\nCAMERA MATRIX (K):\n", K)
print("\nDISTORTION COEFF:\n", dist)

# reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("\n REPROJECTION ERROR:", mean_error / len(objpoints))

# save results
np.save("camera_matrix.npy", K)
np.save("distortion.npy", dist)

print("\n Calibration saved!")