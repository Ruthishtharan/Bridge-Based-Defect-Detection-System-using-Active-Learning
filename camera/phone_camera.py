import cv2


def get_phone_camera(ip: str, timeout_ms: int = 5000):
    """
    Connect to phone IP camera (DroidCam or IP Webcam app).
    Supports formats:
      192.168.1.5:8080      →  http://192.168.1.5:8080/video
      http://192.168.1.5:8080/video  (passed as-is)
    """
    if ip.startswith("http"):
        url = ip
    else:
        url = f"http://{ip}/video"

    print(f"[Camera] Connecting to: {url}")
    # Timeout must be set before opening — use VideoCapture() then open()
    cap = cv2.VideoCapture()
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    cap.open(url)

    if cap.isOpened():
        print("[Camera] ✅ Connected.")
    else:
        print("[Camera] ❌ Connection failed.")

    return cap