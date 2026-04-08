import sys
import os

def print_banner():
    print("\n" + "=" * 50)
    print("  Bridge Defect Detection - Active Learning")
    print("=" * 50)
    print("  0 - Prepare Dataset (SDNET2018)")
    print("  1 - Convert COCO-Bridge Dataset to YOLO")
    print("  2 - Fix Label Filenames")
    print("  3 - Train Model")
    print("  4 - Live Detection (Phone/Webcam)")
    print("  5 - Retrain with Active Learning Samples")
    print("  6 - Label Uncertain Frames (Manual Review)")
    print("  q - Quit")
    print("=" * 50)

def run_option(choice):
    if choice == "0":
        print("\n[->] Preparing SDNET2018 dataset...")
        from utils.dataset_prepare import convert
        convert()
    elif choice == "1":
        print("\n[->] Converting COCO-Bridge dataset to YOLO format...")
        import utils.convert_coco_to_yolo
    elif choice == "2":
        print("\n[->] Fixing label filenames...")
        import utils.fix_label_names
    elif choice == "3":
        print("\n[->] Starting training...")
        from training.train import train
        train()
    elif choice == "4":
        print("\n[->] Starting live detection...")
        from detection.detect_live import run_detection
        run_detection()
    elif choice == "5":
        print("\n[->] Starting active learning retraining...")
        from training.retrain_active import retrain
        retrain()
    elif choice == "6":
        print("\n[->] Launching manual labeling tool...")
        from utils.label_uncertain import label_frames
        label_frames()
    else:
        print("Invalid option. Please select 0-6 or q.")

if __name__ == "__main__":
    while True:
        print_banner()
        choice = input("\nSelect option: ").strip().lower()
        if choice == "q":
            print("Exiting.")
            sys.exit(0)
        run_option(choice)
        input("\n[Press Enter to return to menu...]")
