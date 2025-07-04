# utils/volume_utils.py

import cv2
import math

PIXEL_TO_CM = 0.05

def estimate_volume_cylinder(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, 0.0, 0.0

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    height_cm = h * PIXEL_TO_CM
    diameter_cm = w * PIXEL_TO_CM
    radius_cm = diameter_cm / 2
    volume_cm3 = math.pi * (radius_cm ** 2) * height_cm
    volume_liters = volume_cm3 / 1000

    return volume_liters, height_cm, diameter_cm
