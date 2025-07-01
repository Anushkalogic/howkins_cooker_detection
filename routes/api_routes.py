from flask import Blueprint, jsonify
from database import fetch_all_images_with_volume_in_liters

# Create a Blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Store the most recent detection data
latest_detection = {
    "image_path": None,
    "volume_liters": None,
    "label": None
}

# Live detection (updated during sink function)
@api_bp.route('/live-detection', methods=['GET'])
def get_live_detection():
    return jsonify({
        "success": True,
        "data": {
            "image_path": latest_detection["image_path"],
            "volume_liters": latest_detection["volume_liters"] or 0,
            "label": latest_detection["label"] or "none"
        }
    })

# Database-based detection history
@api_bp.route('/db-detections', methods=['GET'])
def get_db_detections():
    try:
        rows = fetch_all_images_with_volume_in_liters()
        data = []

        for row in rows:
            image_path = row[0]
            volume = int(row[1]) if row[1] is not None else 0
            label = row[2] if row[2] else "none"

            data.append({
                "image_path": image_path,
                "volume_liters": volume,
                "label": label
            })

        return jsonify({
            "success": True,
            "count": len(data),
            "data": data
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Called from my_sink to update latest result
def update_latest_detection(image_path, volume_liters, label):
    latest_detection["image_path"] = image_path
    latest_detection["volume_liters"] = int(volume_liters) if volume_liters is not None else 0
    latest_detection["label"] = label if label else "none"
