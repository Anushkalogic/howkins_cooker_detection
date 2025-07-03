# insert_detection.py
def insert_detection(image_path, volume_liters, label, unique_id, camera, severity):
    global latest_detection
    latest_detection = {
        "image_path": image_path,
        "volume_liters": volume_liters,
        "label": label,
        "unique_id": unique_id,
        "camera_name": camera,
        "severity": severity,
        "height_cm": 0,
        "width_cm": 0
    }
    print(f"LATEST DETECTION SET TO: {latest_detection}")
    # ... DB insert logic ...
