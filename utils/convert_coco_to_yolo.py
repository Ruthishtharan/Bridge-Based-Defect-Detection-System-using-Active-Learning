import os
import shutil

# Auto-detect the dataset root inside the cloned repo
REPO_ROOT   = "data/raw/COCO-Bridge-2021-plus"
DEST_ROOT   = "data/labeled"

# We'll search for the 300x300 folder automatically
def find_300x300(repo_root):
    for dirpath, dirnames, _ in os.walk(repo_root):
        if os.path.basename(dirpath) == "300x300":
            return dirpath
    return None

class_map     = {}
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
        h / img_h,
    )


def process_split(source_root, split_name, dest_name=None):
    global class_counter
    if dest_name is None:
        dest_name = split_name.lower()

    image_src = os.path.join(source_root, split_name, "images")
    bbox_src  = os.path.join(source_root, split_name, "bbox", "txt")

    image_dst = os.path.join(DEST_ROOT, "images", dest_name)
    label_dst = os.path.join(DEST_ROOT, "labels", dest_name)

    os.makedirs(image_dst, exist_ok=True)
    os.makedirs(label_dst, exist_ok=True)

    if not os.path.exists(bbox_src):
        print(f"  WARNING: bbox folder not found: {bbox_src}")
        return

    copied  = 0
    skipped = 0

    for label_file in os.listdir(bbox_src):
        if not label_file.endswith(".txt"):
            continue

        image_name = label_file.replace(".txt", "")
        image_path = os.path.join(image_src, image_name)

        if not os.path.exists(image_path):
            skipped += 1
            continue

        bbox_path = os.path.join(bbox_src, label_file)
        with open(bbox_path, "r") as f:
            lines = f.read().strip().split("\n")

        img_w = int(lines[0])
        img_h = int(lines[1])

        yolo_lines = []
        for line in lines[2:]:
            if not line.strip():
                continue
            parts      = line.split(",")
            class_name = parts[0]
            xmin, ymin, xmax, ymax = map(int, parts[1:5])

            if class_name not in class_map:
                class_map[class_name] = class_counter
                class_counter += 1

            class_id   = class_map[class_name]
            x, y, w, h = convert_bbox(img_w, img_h, xmin, ymin, xmax, ymax)
            yolo_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        with open(os.path.join(label_dst, label_file), "w") as out:
            out.write("\n".join(yolo_lines))

        shutil.copy(image_path, os.path.join(image_dst, image_name))
        copied += 1

    print(f"  {split_name}: {copied} images copied, {skipped} skipped")


def run_conversion():
    source_root = find_300x300(REPO_ROOT)

    if source_root is None:
        print(f"ERROR: Could not find '300x300' folder inside {REPO_ROOT}")
        print("Run: ls data/raw/COCO-Bridge-2021-plus/")
        print("And tell me the folder structure.")
        return

    print(f"[Convert] Found dataset at: {source_root}")
    process_split(source_root, "Train")
    process_split(source_root, "Test", dest_name="val")

    print("\nClass Mapping (copy these into bridge.yaml):")
    for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
        print(f"  {idx}: {name}")

    # Auto-update bridge.yaml with discovered classes
    yaml_path = "data/yaml/bridge.yaml"
    names_block = "\n".join(
        f"  {idx}: {name}"
        for name, idx in sorted(class_map.items(), key=lambda x: x[1])
    )
    yaml_content = f"""path: .

train: data/labeled/images/train
val:   data/labeled/images/val

nc: {len(class_map)}

names:
{names_block}
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nbridge.yaml auto-updated with {len(class_map)} classes.")


# Allow both direct run and import from main.py
if __name__ == "__main__":
    run_conversion()
else:
    run_conversion()
