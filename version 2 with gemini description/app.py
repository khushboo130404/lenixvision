from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import requests
import time
from ultralytics import YOLO
import google.generativeai as genai

# ================= CONFIGURATION =================
app = Flask(__name__)

STATE = {
    "source_base": "",          # ESP32 IP (e.g., http://192.168.1.45)
    "use_stream": True,         # True -> /stream, False -> /capture
    "inference_stride": 3,      # detect every Nth frame
}

# YOLO model
model = YOLO("yolov8n.pt")

# Gemini Vision API
GEMINI_API_KEY = "AIzaSyCnXazKr97QhOoBgGSKCDHfq7RtgXiSXak"  # User provided API key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# File to store latest YOLO detections
LAST_LABEL_FILE = "last_labels.txt"
open(LAST_LABEL_FILE, "w").close()

# ================= HELPER FUNCTIONS =================
def build_urls():
    base = STATE["source_base"].rstrip("/")
    if not base.startswith("http"):
        return None, None
    return f"{base}/stream", f"{base}/capture"

def mjpeg_frames(stream_url):
    with requests.get(stream_url, stream=True, timeout=5) as r:
        r.raise_for_status()
        buf = b""
        for chunk in r.iter_content(chunk_size=1024):
            buf += chunk
            a = buf.find(b'\xff\xd8')
            b = buf.find(b'\xff\xd9')
            if a != -1 and b != -1 and b > a:
                jpg = buf[a:b+2]
                buf = buf[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame

def snapshot_frames(capture_url, fps=8):
    delay = 1.0 / fps
    while True:
        try:
            resp = requests.get(capture_url, timeout=3)
            frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame
        except:
            pass
        time.sleep(delay)

def generate_mjpeg():
    stream_url, capture_url = build_urls()
    if not stream_url:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Set ESP32 IP", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        _, buf = cv2.imencode(".jpg", blank)
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.5)

    frames = mjpeg_frames(stream_url) if STATE["use_stream"] else snapshot_frames(capture_url)

    idx = 0
    for frame in frames:
        idx += 1
        annotated = frame
        detected = []

        if idx % STATE["inference_stride"] == 0:
            try:
                results = model(frame, verbose=False)
                annotated = results[0].plot()
                det = results[0].boxes
                if det is not None and det.cls is not None:
                    names = results[0].names
                    clss = det.cls.cpu().numpy().astype(int)
                    confs = det.conf.cpu().numpy()
                    for c, p in zip(clss, confs):
                        if p > 0.4:
                            detected.append(names.get(c, str(c)))
                if detected:
                    with open(LAST_LABEL_FILE, "w") as f:
                        f.write(",".join(detected))
            except Exception as e:
                print("YOLO error:", e)

        _, buf = cv2.imencode(".jpg", annotated)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

# ================= FLASK ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/labels")
def labels():
    try:
        with open(LAST_LABEL_FILE) as f:
            return jsonify({"labels": f.read()})
    except:
        return jsonify({"labels": ""})

@app.route("/describe", methods=["GET"])
def describe():
    """Send latest frame to Gemini Vision API and return description"""
    stream_url, capture_url = build_urls()
    if not capture_url:
        return jsonify({"text": "ESP32 not set"})

    try:
        resp = requests.get(capture_url, timeout=3)
        frame_bytes = resp.content

        result = gemini_model.generate_content([
            "Describe this scene briefly for a visually impaired person:",
            {"mime_type": "image/jpeg", "data": frame_bytes}
        ])

        text = result.text.strip()
        return jsonify({"text": text})
    except Exception as e:
        print("Gemini error:", e)
        return jsonify({"text": "Error generating description"})

@app.route("/set_source", methods=["POST"])
def set_source():
    data = request.get_json()
    STATE["source_base"] = data.get("base", "")
    STATE["use_stream"] = bool(data.get("use_stream", True))
    return jsonify({"ok": True})

# ================= RUN APP =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
