import torch
from torch.utils.data import DataLoader

from model import UNet
from dataloader import RemoteDataset

# -------------------
# LOAD DATA
# -------------------
train_ds = RemoteDataset("dataset/train/images", "dataset/train/masks")
val_ds   = RemoteDataset("dataset/val/images", "dataset/val/masks")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# -------------------
# MODEL
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------
# TRAIN LOOP
# -------------------
epochs = 6

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f}")

print("Training Completed")

torch.save(model.state_dict(), "unet_remote.pth")
print("Model saved ✔")


