from flask import Flask, render_template, send_file, request, jsonify
import os, cv2, threading, subprocess, sqlite3, csv
from inference import InferencePipeline
from database import init_db, insert_image_with_volume, query_images_by_param

app = Flask(__name__)

# Paths
INPUT_VIDEO = r"static/dataset/videoss.mp4"
OUTPUT_VIDEO = r"static/output/output_video.mp4"
TEMP_OUTPUT = r"static/output/temp_output.mp4"
IMAGE_SAVE_DIR = r"static/detected_images"
DB_PATH = "detections.db"

# Ensure folders exist
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

# Constants
lock = threading.Lock()
frame_count = 0
PIXEL_TO_CM = 0.1  # 10px = 1cm approx
DEPTH_CM = 10   


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT
        )
    ''')

    cursor.execute("PRAGMA table_info(images)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'volume_liters' not in columns:
        cursor.execute("ALTER TABLE images ADD COLUMN volume_liters REAL")
        print("✅ 'volume_liters' column added.")
    else:
        print("ℹ️ 'volume_liters' column already exists.")

    if 'label' not in columns:
        cursor.execute("ALTER TABLE images ADD COLUMN label TEXT")
        print("✅ 'label' column added.")
    else:
        print("ℹ️ 'label' column already exists.")

    conn.commit()
    conn.close()


PIXEL_TO_CM = 0.1  # Adjust based on your setup (e.g. 10 px = 1 cm)
DEPTH_CM = 10      # Assume constant depth if not measurable
def estimate_volume_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        width_cm = w * PIXEL_TO_CM
        height_cm = h * PIXEL_TO_CM
        volume_cm3 = round(width_cm * height_cm * DEPTH_CM, 2)
        volume_liters = round(volume_cm3 / 1000.0, 4)
        return volume_liters
    return 0.0


def insert_image_with_volume(image_path, volume_liters, label):
    image_path = image_path.replace("\\", "/")  # fix for web display
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO images (image_path, volume_liters, label)
        VALUES (?, ?, ?)
    """, (image_path, volume_liters, label))
    conn.commit()
    conn.close()
    print(f"[LOG] Saved to DB -> Image: {image_path}, Volume: {volume_liters} L, Label: {label}")


def fetch_all_images_with_volume_in_liters():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, volume_liters, label FROM images")
    rows = cursor.fetchall()
    conn.close()
    return [(img.replace("\\", "/"), vol, lbl) for img, vol, lbl in rows]


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

    print("🚀 Starting Roboflow inference...")
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

    print("♻️ Re-encoding video...")
    subprocess.call([
        "ffmpeg", "-y", "-i", TEMP_OUTPUT,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        OUTPUT_VIDEO
    ])
    print("✅ Output video saved.")


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

            # Calculate volume in liters
            volume_liters = estimate_volume_from_image(image_path)
            insert_image_with_volume(image_path, volume_liters, "object")
            print(f"📸 {image_path} → Volume: {volume_liters} L")

# ---------- ROUTES ----------
@app.route('/')
def index():
    run_roboflow_pipeline()
    detections = fetch_all_images_with_volume_in_liters()
    return render_template("video_result.html", video_path="output/output_video.mp4", detections=detections)

@app.route('/download-csv')
def download_csv():
    rows = fetch_all_images_with_volume_in_liters()
    csv_path = "static/volume_report.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Path', 'Volume (L)', 'Label'])
        for row in rows:
            image_path, volume_liters, label = row
            writer.writerow([image_path, round(volume_liters, 3), label])
    return send_file(csv_path, as_attachment=True)


def fetch_all_images_with_volume_in_liters():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, volume_liters, label FROM images")
    rows = cursor.fetchall()
    conn.close()
    return rows



@app.route('/query')
def query():
    q = request.args.get("q")
    if not q:
        return jsonify({"error": "Missing ?q=..."}), 400
    results = query_images_by_param(q)
    return jsonify({"results": results})


# ---------- MAIN ----------
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
