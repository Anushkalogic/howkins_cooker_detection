from flask import Flask, render_template
import cv2, os, threading, subprocess
from inference import InferencePipeline
from database import init_db, insert_image_path

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
frame_count = 0

def run_roboflow_pipeline():
    global frame_count
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

        if output_image:
            frame = output_image.numpy_image

            with lock:
                frame_count += 1
                image_path = os.path.join(IMAGE_SAVE_DIR, f"frame_{frame_count}.jpg")

                # Save the frame and write to output video
                cv2.imwrite(image_path, frame)
                out.write(frame)

                # ✅ Only store image path to DB
                insert_image_path(image_path)
                print(f"✅ Stored in DB: {image_path}")

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

    return []  # No label list needed

@app.route('/')
def index():
    run_roboflow_pipeline()
    return render_template("video_result.html", video_path="output/output_video.mp4", detections=[])

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
