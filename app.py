from flask import Flask, render_template, send_file, request, jsonify
import os, cv2, threading, subprocess, csv
from inference import InferencePipeline
from database import init_db, insert_image_with_volume, fetch_all_images_with_volume_in_liters, query_images_by_param

app = Flask(__name__)

INPUT_VIDEO = r"static/dataset/videoss.mp4"
OUTPUT_VIDEO = r"static/output/output_video.mp4"
TEMP_OUTPUT = r"static/output/temp_output.mp4"
IMAGE_SAVE_DIR = r"static/detected_images"


lock = threading.Lock()
frame_count = 0

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

import cv2

PIXEL_TO_CM = 0.2  # Calibrate based on real measurements

# Area thresholds mapped to liter values (in cm²)
volume_thresholds = {
    1: 1000,
    2: 2000,
    3: 3000,
    4: 4000,
    5: 5000,
    6: 6000,
    7: 7000,
    8: 8000,
    9: 9000,
    10: 10000
}

def map_area_to_liters(area_cm2):
    """Map the calculated area (in cm²) to a liter value from 1 to 10."""
    for liter, threshold in volume_thresholds.items():
        if area_cm2 <= threshold:
            return liter
    return 10  # Cap at 10L max

PIXEL_TO_CM = 0.2  # Calibration factor: 1 pixel = 0.2 cm

def estimate_volume_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    area_cm2 = area * PIXEL_TO_CM * PIXEL_TO_CM

    volume_l = area_cm2 / 1000  # 1000 cm³ = 1 L
    volume_l_rounded = round(volume_l, 1)         # for saving or backend
    volume_l_int = int(round(volume_l))           # for image display only

    # Draw integer label on image
    label = f"{volume_l_int} L"
    cv2.putText(image, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Optionally, show area as well
    # area_label = f"{int(area_cm2)} cm²"
    # cv2.putText(image, area_label, (10, 80),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imwrite(image_path, image)

    return volume_l_rounded, "object"

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
def calculate_accuracy(csv_file):
    total = 0
    exact_match = 0
    tolerance_match = 0
    total_error = 0

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        next(reader)  # skip header
        for row in reader:
            try:
                actual = float(row["Actual Volume (L)"])
                predicted = float(row["Volume (L)"].replace(" L", ""))
                total += 1
                error = abs(actual - predicted)
                total_error += error

                if actual == predicted:
                    exact_match += 1

                if error <= 1:
                    tolerance_match += 1
            except:
                continue

    if total == 0:
        print("⚠️ No valid rows to evaluate.")
        return

    exact_accuracy = (exact_match / total) * 100
    tolerance_accuracy = (tolerance_match / total) * 100
    mae = total_error / total

    print("📊 Accuracy Report")
    print(f"Total Samples: {total}")
    print(f"✅ Exact Match Accuracy: {exact_accuracy:.2f}%")
    print(f"✅ ±1L Tolerance Accuracy: {tolerance_accuracy:.2f}%")
    print(f"📉 Mean Absolute Error: {mae:.2f} L")

def my_sink(result, video_frame):
    global frame_count
    output_image = result.get("output_image")

    if output_image:
        frame = output_image.numpy_image
        with lock:
            frame_count += 1
            image_path = os.path.join(IMAGE_SAVE_DIR, f"frame_{frame_count}.jpg")
            cv2.imwrite(image_path, frame)
            out.write(frame)

            volume_liters = estimate_volume_from_image(image_path)
            insert_image_with_volume(image_path, volume_liters, "object")
            print(f"📸 {image_path} → Volume: {volume_liters} L")


@app.route('/')
def index():
    run_roboflow_pipeline()
    detections = fetch_all_images_with_volume_in_liters()
    return render_template("video_result.html", video_path="output/output_video.mp4", detections=detections)

@app.route('/accuracy-report')
def accuracy_report():
    rows = fetch_all_images_with_volume_in_liters()
    
    total = 0
    correct = 0
    for image_path, volume_liters, label in rows:
        # Skip rows with missing or invalid volume
        if volume_liters is None or not isinstance(volume_liters, (int, float)):
            continue
        
        # 👇 ground truth based on filename (e.g., frame_70.jpg -> 7 L expected)
        try:
            frame_name = os.path.basename(image_path)
            frame_number = int(frame_name.split("_")[1].split(".")[0])
            expected_liter = frame_number // 10
        except Exception as e:
            print(f"❌ Failed to parse expected value from {image_path}: {e}")
            continue

        # ✅ Allow +/- 1L margin
        if abs(volume_liters - expected_liter) <= 1:
            correct += 1
        total += 1

    if total == 0:
        print("⚠️ No valid rows to evaluate.")
        return "⚠️ No valid rows to evaluate."

    accuracy = (correct / total) * 100
    print(f"✅ Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
    return f"✅ Accuracy: {accuracy:.2f}% ({correct}/{total} correct)"


@app.route('/download-csv')
def download_csv():
    rows = fetch_all_images_with_volume_in_liters()
    csv_path = "static/volume_report.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Path', 'Volume (L)', 'Label'])

        for row in rows:
            image_path, volume_liters, label = row

            if volume_liters is not None:
                # 🔧 Fix volume formatting — force integer only (1-10)
                volume_liters = min(max(int(volume_liters), 1), 10)
                volume_value = f"{volume_liters} L"
            else:
                volume_value = "N/A"

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
    app.run(debug=True)
