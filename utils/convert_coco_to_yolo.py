import os
import shutil

# -------- CONFIG --------
SOURCE_ROOT = r"data/raw/16624495/COCO-Bridge-2021-plus/COCO-Bridge-2021-plus/300x300"
DEST_ROOT = r"data/labeled"
# ------------------------

class_map = {}
class_counter = 0


def convert_bbox(img_w, img_h, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    return (
        x_center / img_w,
        y_center / img_h,
        w / img_w,
        h / img_h
    )


def process_split(split_name, dest_name=None):
    global class_counter

    if dest_name is None:
        dest_name = split_name.lower()

    image_src = os.path.join(SOURCE_ROOT, split_name, "images")
    bbox_src = os.path.join(SOURCE_ROOT, split_name, "bbox", "txt")

    image_dst = os.path.join(DEST_ROOT, "images", dest_name)
    label_dst = os.path.join(DEST_ROOT, "labels", dest_name)

    os.makedirs(image_dst, exist_ok=True)
    os.makedirs(label_dst, exist_ok=True)

    copied = 0
    skipped = 0

    for label_file in os.listdir(bbox_src):
        if not label_file.endswith(".txt"):
            continue

        # check image exists before doing any work
        image_name = label_file.replace(".txt", "")
        image_path = os.path.join(image_src, image_name)

        if not os.path.exists(image_path):
            skipped += 1
            continue

        # parse bbox annotations
        bbox_path = os.path.join(bbox_src, label_file)

        with open(bbox_path, "r") as f:
            lines = f.read().strip().split("\n")

        img_w = int(lines[0])
        img_h = int(lines[1])

        yolo_lines = []

        for line in lines[2:]:
            parts = line.split(",")
            class_name = parts[0]
            xmin, ymin, xmax, ymax = map(int, parts[1:])

            if class_name not in class_map:
                class_map[class_name] = class_counter
                class_counter += 1

            class_id = class_map[class_name]

            x, y, w, h = convert_bbox(img_w, img_h, xmin, ymin, xmax, ymax)

            yolo_lines.append(f"{class_id} {x} {y} {w} {h}")

        # write YOLO label
        with open(os.path.join(label_dst, label_file), "w") as out:
            out.write("\n".join(yolo_lines))

        # copy image
        shutil.copy(image_path, os.path.join(image_dst, image_name))
        copied += 1

    print(f"{split_name} Done. ({copied} images copied, {skipped} skipped)")


# Process Train and Test (Test → val to match bridge.yaml)
process_split("Train")
process_split("Test", dest_name="val")

print("\nClass Mapping:")
for k, v in class_map.items():
    print(f"  {v}: {k}")
