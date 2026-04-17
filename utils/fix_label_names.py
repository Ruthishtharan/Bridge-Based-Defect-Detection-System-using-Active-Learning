import os

label_root = "data/labeled/labels"


def fix_label_names():
    for split in ["train", "val"]:
        folder = os.path.join(label_root, split)

        if not os.path.exists(folder):
            print(f"[Warning] Label folder not found, skipping: {folder}")
            continue

        for filename in os.listdir(folder):
            if filename.endswith(".jpeg.txt"):
                old_path = os.path.join(folder, filename)
                new_name = filename.replace(".jpeg.txt", ".txt")
                new_path = os.path.join(folder, new_name)
                os.rename(old_path, new_path)

    print("Label renaming complete.")


if __name__ == "__main__":
    fix_label_names()
