import os
from PIL import Image

CLASS_MAP = {
    "Crack": 0,
    "Bearing": 1,
    "Spall": 2,
    "Rebar": 3
}

BASE_ORIG = "data/raw/coco_bridge/COCO-Bridge-2021-plus/original"
BASE_OUT = "data/raw/coco_bridge/COCO-Bridge-2021-plus/300x300"

for split in ["Train", "Test"]:
    src_dir = os.path.join(BASE_ORIG, split, "bbox", "txt")
    img_dir = os.path.join(BASE_ORIG, split, "images")
    dst_dir = os.path.join(BASE_OUT, split, "labels")

    for file in os.listdir(src_dir):
        if not file.endswith(".txt"):
            continue

        src_path = os.path.join(src_dir, file)
        img_name = file.replace(".txt", "")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        dst_name = file.replace(".jpeg.txt", ".txt")
        dst_path = os.path.join(dst_dir, dst_name)

        new_lines = []

        with open(src_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(",")

            if len(parts) != 5:
                continue

            cls_name = parts[0]
            if cls_name not in CLASS_MAP:
                continue

            cls = CLASS_MAP[cls_name]
            x1, y1, x2, y2 = map(float, parts[1:5])

            # correct normalization
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            new_lines.append(f"{cls} {x_center} {y_center} {width} {height}\n")

        with open(dst_path, "w") as f:
            f.writelines(new_lines)

print("✅ Labels FIXED with correct normalization")
