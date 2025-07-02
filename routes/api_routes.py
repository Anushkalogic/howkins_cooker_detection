from flask import Blueprint, jsonify
from database import fetch_all_images_with_volume_in_liters
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Store the most recent detection data
latest_detection = {
    "image_path": None,
    "volume_liters": None,
    "label": None,
    "unique_id": None  # 👈 added this line
}

# Live detection (updated during sink function)
@api_bp.route('/live-detection', methods=['GET'])
def get_live_detection():
    return jsonify({
        "success": True,
        "data": {
            "image_path": latest_detection["image_path"],
            "volume_liters": latest_detection["volume_liters"] or 0,
            "defect": latest_detection["label"] or "none",  # label → defect
            "unique_id": latest_detection["unique_id"] or "N/A"
        }
    })


@api_bp.route('/db-detections', methods=['GET'])
def get_db_detections():
    try:
        rows = fetch_all_images_with_volume_in_liters()
        data = []

        for row in rows:
            image_path = row[0]
            volume = int(row[1]) if row[1] is not None else 0
            defect = row[2] if row[2] else "object"  # label renamed to defect
            unique_id = row[3] if row[3] else "N/A"

            data.append({
                "image_path": image_path,
                "volume_liters": volume,
                "defect": defect,
                "unique_id": unique_id  # ✅ use the actual ID
            })

        return jsonify({
            "success": True,
            "count": len(data),
            "data": data
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Called from my_sink to update latest result
def update_latest_detection(image_path, volume_liters, label, unique_id):
    latest_detection["image_path"] = image_path
    latest_detection["volume_liters"] = int(volume_liters) if volume_liters is not None else 0
    latest_detection["label"] = label if label else "none"
    latest_detection["unique_id"] = unique_id  # 👈 store ID too


    
