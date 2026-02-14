import os
import shutil
import random

SOURCE_DIR = "data/raw/DATA_Maguire_20180517_ALL/SDNET2018"
DEST_DIR = "data/labeled"

TRAIN_SPLIT = 0.8

# Folder prefixes that indicate cracked images
CRACKED_PREFIXES = ("CD", "CW", "CP")


def create_dirs():
    dirs = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val"
    ]

    for d in dirs:
        os.makedirs(os.path.join(DEST_DIR, d), exist_ok=True)


def is_cracked(img_path):
    """Determine if image is cracked based on its parent folder name."""
    parent_folder = os.path.basename(os.path.dirname(img_path))
    return parent_folder.startswith(CRACKED_PREFIXES)


def convert():
    create_dirs()

    all_images = []

    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(".jpg"):
                all_images.append(os.path.join(root, file))

    if not all_images:
        print(f"No images found in {SOURCE_DIR}. Check the path.")
        return

    random.shuffle(all_images)

    split_index = int(len(all_images) * TRAIN_SPLIT)

    train = all_images[:split_index]
    val = all_images[split_index:]

    process_images(train, "train")
    process_images(val, "val")

    cracked_count = sum(1 for img in all_images if is_cracked(img))
    print(f"Dataset prepared: {len(all_images)} images "
          f"({cracked_count} cracked, {len(all_images) - cracked_count} uncracked)")
    print(f"  Train: {len(train)}, Val: {len(val)}")


def process_images(images, split):

    for img_path in images:

        filename = os.path.basename(img_path)

        dest_img = os.path.join(
            DEST_DIR,
            f"images/{split}/{filename}"
        )

        shutil.copy(img_path, dest_img)

        label_name = filename.replace(".jpg", ".txt")

        label_path = os.path.join(
            DEST_DIR,
            f"labels/{split}/{label_name}"
        )

        with open(label_path, "w") as f:
            if is_cracked(img_path):
                # class 0 = crack, full-image bounding box
                f.write("0 0.5 0.5 1.0 1.0\n")
            # else: empty label file = no detections (background/uncracked)


if __name__ == "__main__":
    convert()
