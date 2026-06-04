# Measurement Report

## Methodology
The system computes real-world object dimensions using the following pipeline:

1. Image undistortion using camera intrinsic parameters
2. Segmentation using trained UNet model
3. Contour extraction from binary mask
4. Bounding box computation
5. Conversion from pixels to millimetres using calibration scale

---

## Scale Factor
1 pixel = 0.622 mm

---

## Sample Result

Example output:
- Width: 355.5 mm
- Height: 181.75 mm

---

## Error Analysis
Minor deviations may occur due to:
- Segmentation boundary noise
- Perspective distortion
- Object placement variation

---

## Conclusion
The system successfully converts image-based measurements into real-world units with reasonable accuracy.
