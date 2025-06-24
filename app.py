from flask import Flask, render_template
import cv2, os, subprocess, threading
from inference import InferencePipeline

app = Flask(__name__)

# Paths
INPUT_VIDEO = r"static/dataset/video.mp4"
TEMP_OUTPUT = r"static/temp_output.mp4"
OUTPUT_VIDEO = r"static/output.mp4"

# Shared objects
lock = threading.Lock()
all_detections = []

def run_roboflow_pipeline():
    global all_detections

    # Clear previous
    all_detections.clear()

    # Load original video to get dimensions
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open input video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Set up temp video writer
    out = cv2.VideoWriter(TEMP_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Callback per frame
    def my_sink(result, video_frame):
        nonlocal out
        if result.get("output_image"):
            frame = result["output_image"].numpy_image
            with lock:
                out.write(frame)
            for pred in result.get("predictions", []):
                label = pred.get("class")
                confidence = pred.get("confidence", 0)
                all_detections.append(f"{label} ({confidence:.0%})")

    # Start inference
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

    with lock:
        out.release()

    # ✅ Re-encode using FFmpeg to ensure compatibility
    print("♻️ Re-encoding video to playable format...")
    subprocess.call([
        "ffmpeg", "-y", "-i", TEMP_OUTPUT,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        OUTPUT_VIDEO
    ])
    print("✅ Video re-encoding complete.")

    return list(dict.fromkeys(all_detections))  # Remove duplicates

@app.route('/')
def index():
    detections = run_roboflow_pipeline()
    return render_template("video_result.html", video_path="output.mp4", detections=detections)

if __name__ == '__main__':
    app.run(debug=True)
