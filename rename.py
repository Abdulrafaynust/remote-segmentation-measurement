import os

folder = "calibration"  # your folder name

files = sorted(os.listdir(folder))

for i, file in enumerate(files):
    ext = os.path.splitext(file)[1]  # keep .jpg/.png

    new_name = f"img_{i+1:03d}{ext}"  # img_001.jpg, img_002.jpg ...

    old_path = os.path.join(folder, file)
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)

print("Renaming done ✔")