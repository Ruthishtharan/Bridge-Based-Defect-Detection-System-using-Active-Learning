import os

label_root = "data/labeled/labels"

for split in ["train", "val"]:
    folder = os.path.join(label_root, split)

    for filename in os.listdir(folder):
        if filename.endswith(".jpeg.txt"):
            old_path = os.path.join(folder, filename)
            new_name = filename.replace(".jpeg.txt", ".txt")
            new_path = os.path.join(folder, new_name)

            os.rename(old_path, new_path)

print("Label renaming complete.")
