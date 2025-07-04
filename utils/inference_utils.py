from utils.volume_utils import estimate_volume_cylinder

def run_inference_single_frame(frame):
    volume_liters, height_cm, width_cm = estimate_volume_cylinder(frame)

    label = "object"
    severity = "None"

    return {
        "volume_liters": int(volume_liters),
        "height_cm": round(height_cm, 1),
        "width_cm": round(width_cm, 1),
        "label": label,
        "severity": severity
    }
