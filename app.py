from flask import Flask, render_template
import cv2, os, threading, hashlib, subprocess
from inference import InferencePipeline
from database import init_db, insert_detection

app = Flask(__name__)

# Paths
INPUT_VIDEO = r"static/dataset/videoss.mp4"
OUTPUT_VIDEO = r"static/output/output_video.mp4"
TEMP_OUTPUT = r"static/output/temp_output.mp4"
IMAGE_SAVE_DIR = r"static/detected_images"

# Ensure folders exist
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

lock = threading.Lock()
all_detections = []
frame_count = 0

def hash_frame(frame):
    return hashlib.md5(cv2.imencode('.jpg', frame)[1]).hexdigest()

def run_roboflow_pipeline():
    global all_detections, frame_count
    all_detections.clear()
    frame_count = 0

    # Clean previous frames
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

    def my_sink(result, video_frame):
        global frame_count
        output_image = result.get("output_image")
        predictions = result.get("predictions", [])

        if output_image and predictions:
            frame = output_image.numpy_image
            with lock:
                out.write(frame)

            frame_count += 1
            image_path = os.path.join(IMAGE_SAVE_DIR, f"frame_{frame_count}.jpg")
            cv2.imwrite(image_path, frame)

            for pred in predictions:
                label = pred.get("class")
                confidence = pred.get("confidence", 0)
                insert_detection(label, confidence, image_path)
                all_detections.append(f"{label} ({confidence:.0%})")

    print("🚀 Starting Roboflow inference...")
    pipeline = InferencePipeline.init_with_workflow(
        api_key="DiBsOHUZVRTHIOZjUoWJ",
        workspace_name="anushka-t2wnn",
        workflow_id="detect-count-and-visualize",
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

    return list(dict.fromkeys(all_detections))  # Remove duplicates

@app.route('/')
def index():
    detections = run_roboflow_pipeline()
    return render_template("video_result.html", video_path="output/output_video.mp4", detections=detections)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
