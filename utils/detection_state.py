# utils/detection_state.py

latest_detection = {}

def update_latest_detection(data):
    global latest_detection
    latest_detection = data
