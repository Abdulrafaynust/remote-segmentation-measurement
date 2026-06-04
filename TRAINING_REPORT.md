# Training Report

## Model Architecture
UNet (custom implementation in PyTorch)

---

## Training Configuration
- Epochs: 6
- Batch Size: 4
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Binary Cross Entropy (BCELoss)

---

## Results

Final Training Loss:
0.1696

Training shows consistent convergence across epochs.

---

## Observations
- Model successfully learns object boundaries
- Stable loss reduction observed
- Performs well on unseen validation images

---

## Conclusion
The UNet model provides reliable segmentation results for downstream measurement tasks.
