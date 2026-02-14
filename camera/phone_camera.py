import cv2

def get_phone_camera(ip):
    url = f"http://{ip}/video"
    cap = cv2.VideoCapture(url)
    return cap
