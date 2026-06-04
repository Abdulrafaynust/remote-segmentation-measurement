import torch
import cv2
import numpy as np
from model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load("unet_remote.pth", map_location=device))
model.eval()

IMG_SIZE = 256

def predict(image_path):
    img = cv2.imread(image_path)
    orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)[0][0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
    
    cv2.imwrite("mask.png", mask)
    print("Mask saved")

    cv2.imshow("Mask", mask)
    cv2.imshow("Original", orig)
    cv2.waitKey(0)

# TEST
predict("test.jpeg")