print("Bridge Defect Active Learning System")
print("=" * 40)
print("0 - Prepare Dataset")
print("1 - Train Model")
print("2 - Live Detection")
print("3 - Retrain using Active Learning")

choice = input("Select option: ")

if choice == "0":
    from utils.dataset_prepare import convert
    convert()
elif choice == "1":
    import training.train
elif choice == "2":
    import detection.detect_live
elif choice == "3":
    import training.retrain_active
else:
    print("Invalid option. Please select 0-3.")
