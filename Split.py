import os
import random
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))

print("BASE DIR:", base_dir)

img_dir = os.path.join(base_dir, "dataset", "images")
mask_dir = os.path.join(base_dir, "dataset", "masks")

print("Image dir:", img_dir)
print("Mask dir:", mask_dir)

# Create folders properly
splits = ["train", "val", "test"]

for s in splits:
    os.makedirs(os.path.join(base_dir, "dataset", s, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "dataset", s, "masks"), exist_ok=True)

print("Folders created")

# Check images exist
images = sorted(os.listdir(img_dir))
print("Total images:", len(images))

random.shuffle(images)

n = len(images)
train_end = int(0.7 * n)
val_end = int(0.9 * n)

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]

def copy_data(img_list, split):
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.rsplit(".",1)[0] + ".png")

        shutil.copy(img_path, os.path.join(base_dir, "dataset", split, "images"))
        shutil.copy(mask_path, os.path.join(base_dir, "dataset", split, "masks"))

copy_data(train_imgs, "train")
copy_data(val_imgs, "val")
copy_data(test_imgs, "test")

print("Dataset split completed ")