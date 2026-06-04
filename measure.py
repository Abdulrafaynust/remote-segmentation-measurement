import cv2
import numpy as np

def measure_object(mask_path, pixel_to_mm=0.5):
    # Load mask
    mask = cv2.imread(mask_path, 0)

    if mask is None:
        print("❌ Mask not found:", mask_path)
        return

    # Ensure binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("❌ No object detected in mask")
        return

    # Get largest contour (TV remote)
    cnt = max(contours, key=cv2.contourArea)

    # Bounding box
    x, y, w, h = cv2.boundingRect(cnt)

    # Convert to mm
    width_mm = w * pixel_to_mm
    height_mm = h * pixel_to_mm

    # Print results
    print("\n📏 MEASUREMENT RESULTS")
    print("------------------------")
    print("Width (pixels):", w)
    print("Height (pixels):", h)
    print("Width (mm):", round(width_mm, 2))
    print("Height (mm):", round(height_mm, 2))

    # Visualization
    img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Measurement", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------
# RUN TEST
# -------------------
measure_object("mask.png", pixel_to_mm=0.5)