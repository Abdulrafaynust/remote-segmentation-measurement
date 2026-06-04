import cv2
import numpy as np

# CHANGE THIS based on your checkerboard
SQUARE_SIZE_MM = 25  # e.g. 25mm per square

def compute_scale(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect corners
    pattern_size = (7, 7)  # adjust if your checkerboard differs
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if not ret:
        print(" Checkerboard not detected")
        return None

    # take first two adjacent corners
    p1 = corners[0][0]
    p2 = corners[1][0]

    pixel_dist = np.linalg.norm(p1 - p2)

    pixel_to_mm = SQUARE_SIZE_MM / pixel_dist

    print("Pixel distance:", pixel_dist)
    print("Pixel → MM scale:", pixel_to_mm)

    return pixel_to_mm


# TEST
scale = compute_scale("test.jpeg")
print("FINAL SCALE:", scale)