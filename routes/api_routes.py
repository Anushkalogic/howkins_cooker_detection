from flask import Blueprint, jsonify
from database import fetch_all_images_with_volume_in_liters

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Store the most recent detection data
latest_detection = {
    "image_path": None,
    "volume_liters": None,
    "label": None,
    "id": None,              # 🔁 changed from "unique_id"
    "camera": None,          # 🔁 changed from "camera_name"
    "severity": None         # ✅ add severity
}

# Live detection (updated during sink function)
@api_bp.route('/live-detection', methods=['GET'])
def get_live_detection():
    return jsonify({
        "success": True,
        "data": {
            "image_path": latest_detection.get("image_path"),
            "volume_liters": latest_detection.get("volume_liters", 0),
            "defect": latest_detection.get("label", "none"),
            "id": latest_detection.get("id", "N/A"),          # ✅ corrected
            "camera": latest_detection.get("camera", "N/A"),  # ✅ corrected
            "height_cm": latest_detection.get("height_cm", 0),
            "width_cm": latest_detection.get("width_cm", 0),
            "severity": latest_detection.get("severity", "None")
        }
    })




@api_bp.route('/db-detections', methods=['GET'])
def get_db_detections():
    try:
        rows = fetch_all_images_with_volume_in_liters()
        data = []

        for row in rows:
            image_path = row[0]
            volume = float(row[1]) if row[1] is not None else 0.0
            defect = row[2] if row[2] else "object"
            unique_id = row[3] if row[3] else "N/A"
            camera_name = row[4] if len(row) > 4 and row[4] else "N/A"

            # 🔥 Determine severity based on defect type and volume
           # 🔥 Custom severity logic based on defect type
            if defect.lower() == "full":
                severity = "High"
            elif defect.lower() == "dent":
                severity = "Medium"
            elif defect.lower() == "scrach":
                severity = "Low"
            else:
                severity = "None"

            data.append({
                "image_path": image_path,
                "volume_liters": f"{int(volume)} L" if volume else "0 L",
                "defect": defect,
                "id": unique_id,               # ⬅️ Renamed
                "camera": camera_name,         # ⬅️ Renamed
                "severity": severity
            })

        return jsonify({
            "success": True,
            "count": len(data),
            "data": data
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Called from my_sink to update latest result
# ✅ At the bottom of routes/api_routes.py
def update_latest_detection(image_path, volume_liters, label, unique_id, camera_name, height_cm, width_cm, severity):
    latest_detection["image_path"] = image_path
    latest_detection["volume_liters"] = int(volume_liters) if volume_liters is not None else 0
    latest_detection["label"] = label if label else "none"
    latest_detection["id"] = unique_id
    latest_detection["camera"] = camera_name
    latest_detection["severity"] = severity
    latest_detection["height_cm"] = round(height_cm, 1)
    latest_detection["width_cm"] = round(width_cm, 1)

      # ⬅️ Renamed
