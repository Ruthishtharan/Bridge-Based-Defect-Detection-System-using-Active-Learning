import os

IMG_SIZE = 300

CLASS_MAP = {
    "Crack": 0,
    "Bearing": 1,
    "Spall": 2,
    "Rebar": 3
}

BASE_ORIG = "data/raw/coco_bridge/COCO-Bridge-2021-plus/original"
BASE_OUT = "data/raw/coco_bridge/COCO-Bridge-2021-plus/300x300"


def convert_coco_labels():
    """Convert COCO-Bridge labels to YOLO format (300x300 normalized)."""
    for split in ["Train", "Test"]:
        src_dir = os.path.join(BASE_ORIG, split, "bbox", "txt")
        dst_dir = os.path.join(BASE_OUT, split, "labels")

        if not os.path.exists(src_dir):
            print(f"[Warning] Label directory not found, skipping: {src_dir}")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        for file in os.listdir(src_dir):
            if not file.endswith(".txt"):
                continue

            src_path = os.path.join(src_dir, file)
            dst_name = file.replace(".jpeg.txt", ".txt")
            dst_path = os.path.join(dst_dir, dst_name)

            new_lines = []

            with open(src_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split(",")

                if len(parts) != 5:
                    print(f"[Warning] Skipping malformed line in {file}: {line.strip()}")
                    continue

                cls_name = parts[0]

                if cls_name not in CLASS_MAP:
                    continue

                cls = CLASS_MAP[cls_name]
                try:
                    x1, y1, x2, y2 = map(float, parts[1:5])
                except ValueError:
                    print(f"[Warning] Non-numeric coordinates in {file}: {line.strip()}")
                    continue

                x_center = max(0.0, min(1.0, ((x1 + x2) / 2) / IMG_SIZE))
                y_center = max(0.0, min(1.0, ((y1 + y2) / 2) / IMG_SIZE))
                width    = max(0.0, min(1.0, (x2 - x1) / IMG_SIZE))
                height   = max(0.0, min(1.0, (y2 - y1) / IMG_SIZE))

                new_lines.append(f"{cls} {x_center} {y_center} {width} {height}\n")

            with open(dst_path, "w") as f:
                f.writelines(new_lines)

    print("✅ Labels rebuilt correctly from original dataset")
