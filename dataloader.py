import os
import cv2
import torch
from torch.utils.data import Dataset

IMG_SIZE = 256

class RemoteDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name.rsplit(".",1)[0] + ".png")

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Read mask
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        # Normalize
        image = image / 255.0
        mask = mask / 255.0

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask