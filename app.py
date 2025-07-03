from flask import Flask, render_template, send_file, request, jsonify
import os, cv2, threading, subprocess, csv, math
from inference import InferencePipeline
from routes.api_routes import update_latest_detection  # 👈 import
import numpy as np
from database import init_db, insert_image_with_volume, fetch_all_images_with_volume_in_liters, query_images_by_param,update_defect_entries
from database import cleanup_null_entries
from routes.api_routes import api_bp  # 👈 yaha pura blueprint import karo

app = Flask(__name__)
app.register_blueprint(api_bp)  # 👈 ye register tab karo jab pura define ho chuka ho
debug=True

INPUT_VIDEO = r"static/dataset/videoss.mp4"
OUTPUT_VIDEO = r"static/output/output_video.mp4"
TEMP_OUTPUT = r"static/output/temp_output.mp4"
IMAGE_SAVE_DIR = r"static/detected_images"
import cv2
import math

lock = threading.Lock()
frame_count = 0

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)



PIXEL_TO_CM = 0.05  

def run_roboflow_pipeline():
    global frame_count, out
    frame_count = 0
    for f in os.listdir(IMAGE_SAVE_DIR):
        os.remove(os.path.join(IMAGE_SAVE_DIR, f))

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open input video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out = cv2.VideoWriter(TEMP_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    pipeline = InferencePipeline.init_with_workflow(
        api_key="DiBsOHUZVRTHIOZjUoWJ",
        workspace_name="anushka-t2wnn",
        workflow_id="detect-count-and-visualize-5",
        video_reference=INPUT_VIDEO,
        max_fps=30,
        on_prediction=my_sink
    )

    pipeline.start()
    pipeline.join()
    out.release()

    subprocess.call([
        "ffmpeg", "-y", "-i", TEMP_OUTPUT,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        OUTPUT_VIDEO
    ])

    cleanup_null_entries()
def my_sink(result, video_frame):
    global frame_count
    output_image = result.get("output_image")
    predictions = result.get("predictions", [])

    if output_image:
        frame = output_image.numpy_image
        with lock:
            frame_count += 1
            image_path = os.path.join(IMAGE_SAVE_DIR, f"frame_{frame_count}.jpg")

            # Estimate volume + dimensions
            volume_liters, height_cm, diameter_cm = estimate_volume_cylinder(frame)
            volume_liters = int(round(volume_liters)) if volume_liters is not None else 0  # ✅ Integer only

            # Get label
            label = "object"
            if hasattr(predictions, "data") and "class_name" in predictions.data:
                class_names = predictions.data["class_name"]
                if isinstance(class_names, np.ndarray) and len(class_names) > 0:
                    label = ", ".join(class_names.tolist()).strip()
                    if not label:
                        label = "object"

            # Determine severity
            if "pull" in label.lower():
                severity = "High"
            elif "dent" in label.lower():
                severity = "Medium"
            elif "scratch" in label.lower():
                severity = "Low"
            else:
                severity = "None"

            # 👉 Draw bounding boxes WITHOUT label
            if hasattr(predictions, "xyxy") and hasattr(predictions, "class_name"):
                boxes = predictions.xyxy
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Purple box

            # 👉 Draw text on side (no box label)
            text_color = (255, 255, 255)  # White
            text_lines = [
                f"Height: {round(height_cm, 1)} cm",
                f"Width : {round(diameter_cm, 1)} cm",
                f"Volume: {volume_liters} L",  # ✅ integer L
                f"Defect: {label}",
                # f"Defect: {label}",
                # f"Severity: {severity}"

            ]

            for i, line in enumerate(text_lines):
                y = 30 + i * 30
                # cv2.rectangle(frame, (5, y - 25), (310, y + 5), (0, 0, 0), -1)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            # Save frame
            cv2.imwrite(image_path, frame)
            out.write(frame)

            # Save to DB
            camera_name = "1"
            unique_id, camera_name = insert_image_with_volume(
                image_path.replace("\\", "/"), volume_liters, label, camera_name
            )
            update_latest_detection(
                image_path.replace("\\", "/"), volume_liters, label, unique_id, camera_name
            )

            print(f"📸 {image_path} → Volume: {volume_liters} L | Label: {label}")
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

# @app.route('/')
# def index():
#     run_roboflow_pipeline()
#     detections = fetch_all_images_with_volume_in_liters()
#     return render_template("video_result.html", video_path="output/output_video.mp4", detections=detections)


@app.route('/')
def index():
    run_roboflow_pipeline()
    detections = fetch_all_images_with_volume_in_liters()

    total_frames = len(detections)
    dented = sum(1 for _, _, label, _, _ in detections if 'dent' in str(label).lower())
    scratched = sum(1 for _, _, label, _, _ in detections if 'scratch' in str(label).lower())


    return render_template(
        "video_result.html",
        video_path="output/output_video.mp4",
        detections=detections,
        total_frames=total_frames,
        dented=dented,
        scratched=scratched
    )


@app.route('/get-live-count')
def get_live_count():
    return jsonify({"count": frame_count})


@app.route('/show-db')
def show_db():
    from database import fetch_all_images_with_volume_in_liters
    data = fetch_all_images_with_volume_in_liters()
    return jsonify(data)
# delete all entry from db use as per your requirement
@app.route('/delete-all')
def delete_all_entries():
    import sqlite3
    from database import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM images")  # 🚨 This deletes all records!
    conn.commit()
    conn.close()

    return "✅ All entries deleted from DB."

@app.route('/download-csv')
def download_csv():
    rows = fetch_all_images_with_volume_in_liters()
    csv_path = "static/volume_report.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Path', 'Volume (L)', 'Label'])

        for row in rows:
            image_path, volume_liters, label = row
            volume_value = f"{int(volume_liters)} L" if volume_liters is not None else "N/A"
            writer.writerow([image_path, volume_value, label])

    return send_file(csv_path, as_attachment=True)



@app.route('/query')
def query():
    q = request.args.get("q")
    if not q:
        return jsonify({"error": "Missing ?q=..."}), 400
    results = query_images_by_param(q)
    return jsonify({"results": results})

if __name__ == '__main__':
    init_db()
    update_defect_entries()
    app.run(debug=True)
