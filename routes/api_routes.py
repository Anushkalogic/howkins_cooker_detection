from flask import Blueprint, jsonify
from database import fetch_all_images_with_volume_in_liters

api_bp = Blueprint('api', __name__, url_prefix='/api')

latest_detection = {
    "image_path": None,
    "volume_liters": None,
    "label": None
}

@api_bp.route('/live-detection', methods=['GET'])
def get_live_detection():
    return jsonify(latest_detection)

@api_bp.route('/db-detections', methods=['GET'])
def get_db_detections():
    rows = fetch_all_images_with_volume_in_liters()
    return jsonify([
        {
            "image_path": row[0],
            "volume_liters": row[1],
            "label": row[2]
        } for row in rows
    ])

def update_latest_detection(image_path, volume_liters, label):
    latest_detection["image_path"] = image_path
    latest_detection["volume_liters"] = volume_liters
    latest_detection["label"] = label
