"""
app.py — Smart Classroom AI Dashboard Backend
Flask + Socket.IO server: MJPEG stream, REST API, real-time events
"""

import os, sys, time, threading, base64

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, make_response, send_from_directory
from flask_socketio import SocketIO
import csv
import datetime
from io import StringIO

# ── Resolve local imports ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector.yolo_detector import YOLODetector
from utils.visualizer import Visualizer
from config import MODEL_PATH, CONF_THRESHOLD, CAMERA_INDEX

from db.database import (
    init_db, start_session, end_session, record_frame, flush_all,
    get_sessions, get_session_frames, get_analytics, get_hourly_summary, get_session_timeline
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "smart-classroom-2025"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Load model & init DB at startup ───────────────────────────────────────────
print("[APP] Loading YOLO model …")
detector   = YOLODetector(MODEL_PATH, CONF_THRESHOLD)
visualizer = Visualizer()
print("[APP] Model ready.")

print("[APP] Initializing Database …")
init_db()

# ── Shared mutable state ──────────────────────────────────────────────────────
state = {
    "running"      : False,
    "source_type"  : "idle",
    "session_id"   : None,
    "frame_count"  : 0,
    "fps"          : 0.0,
    "total_persons": 0,
    "quadrants"    : {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0},
    "history"      : [],          # rolling list of frame records
}

output_frame  = None              # latest JPEG bytes for MJPEG stream
frame_lock    = threading.Lock()
stop_event    = threading.Event()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_quadrant_counts(frame, detections):
    h, w    = frame.shape[:2]
    mx, my  = w // 2, h // 2
    counts  = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for det in detections:
        if det["label"].lower() != "person":
            continue
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if   cx < mx and cy < my:  counts["Q1"] += 1
        elif cx >= mx and cy < my: counts["Q2"] += 1
        elif cx < mx and cy >= my: counts["Q3"] += 1
        else:                      counts["Q4"] += 1
    return counts

def get_appliances_state(total):
    """Energy-saving rules: decide fan/light count and calculate wattage savings."""
    max_w = (6 * 75) + (8 * 30)  # 690W total ceiling
    if total == 0:
        fans, lights = 0, 0
        mode, level = "ECO SAVING (0 Students)", "eco"
    elif total <= 10:
        fans, lights = 2, 4
        mode, level = "LOW POWER (1-10 Students)", "low"
    elif total < 20:
        fans, lights = 4, 6
        mode, level = "MEDIUM POWER (11-19 Students)", "med"
    else:
        fans, lights = 6, 8
        mode, level = "MAX POWER (20+ Students)", "high"

    load_w = (fans * 75) + (lights * 30)
    saved_w = max_w - load_w
    savings_pct = round((saved_w / max_w) * 100, 1)

    return {
        "fans": fans,
        "lights": lights,
        "mode": mode,
        "level": level,
        "load_w": load_w,
        "saved_w": saved_w,
        "savings_pct": savings_pct
    }


def point_quadrant(frame, cx, cy):
    h, w = frame.shape[:2]
    if cx < w // 2 and cy < h // 2: return "Q1"
    if cx >= w // 2 and cy < h // 2: return "Q2"
    if cx < w // 2 and cy >= h // 2: return "Q3"
    return "Q4"


# ── Detection loop (runs in background thread) ────────────────────────────────
def detection_loop(source):
    global output_frame

    cap = cv2.VideoCapture(source)
    if not cap.isOpened() and (source == CAMERA_INDEX or source == 0 or source == "0"):
        # Graceful fallback to multi-scene classroom demo video
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "sample.mp4")
        if os.path.exists(sample_path):
            cap = cv2.VideoCapture(sample_path)
            source = sample_path
            socketio.emit("srv_notice", {
                "message": "Webcam 0 unavailable. Auto-switched to multi-scene classroom demo feed."
            })

    if not cap.isOpened():
        state["running"] = False
        socketio.emit("srv_error", {"message": f"Cannot open source: {source}"})
        return

    prev_time   = time.time()
    frame_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if not isinstance(source, int):   # loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.03)
                continue
            break

        now       = time.time()
        dt        = now - prev_time
        fps       = 1.0 / dt if dt > 0 else 0.0
        prev_time = now
        frame_count += 1

        detections = detector.detect(frame)
        counts     = get_quadrant_counts(frame, detections)
        total      = sum(counts.values())

        # Annotate
        annotated  = visualizer.draw(frame.copy(), detections)
        visualizer.draw_quadrants(annotated, counts)

        # Update shared state
        state.update({
            "frame_count"  : frame_count,
            "fps"          : round(fps, 1),
            "total_persons": total,
            "quadrants"    : counts,
        })
        record = {
            "ts"   : round(now * 1000),
            "frame": frame_count,
            "fps"  : round(fps, 1),
            **counts,
            "total": total,
        }
        state["history"].append(record)
        if len(state["history"]) > 500:
            state["history"].pop(0)

        # Record to database
        if state["session_id"] is not None:
            record_frame(state["session_id"], frame_count, fps,
                         counts["Q1"], counts["Q2"], counts["Q3"], counts["Q4"])

        # Encode JPEG for MJPEG stream
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
        with frame_lock:
            output_frame = buf.tobytes()

        # Push real-time update to all connected clients
        socketio.emit("update", {
            "frame": frame_count,
            "fps"  : round(fps, 1),
            "Q1"   : counts["Q1"],
            "Q2"   : counts["Q2"],
            "Q3"   : counts["Q3"],
            "Q4"   : counts["Q4"],
            "total": total,
            "appliances": get_appliances_state(total)
        })

        # Cap speed for video files to realistic 24-30 fps
        if not isinstance(source, int):
            elapsed = time.time() - now
            sleep_time = 0.04 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()
    
    if state["session_id"] is not None:
        flush_all()
        end_session(state["session_id"])
        state["session_id"] = None

    state["running"] = False
    with frame_lock:
        output_frame = None
    socketio.emit("srv_stopped", {})


# ── MJPEG generator ───────────────────────────────────────────────────────────
def mjpeg_stream():
    while True:
        with frame_lock:
            data = output_frame
        if data is None:
            time.sleep(0.04)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.033)   # cap at ~30 fps pushed to client


# ── Core Streaming Routes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Stream Lifecycle Controls ─────────────────────────────────────────────────
@app.route("/api/stream/start", methods=["POST"])
@app.route("/api/start", methods=["POST"])
def api_start():
    data        = request.json or {}
    source_type = data.get("source", "webcam")

    if state["running"]:
        return jsonify({"error": "Detection already running", "code": "ALREADY_RUNNING"}), 400

    if source_type == "webcam":
        source = CAMERA_INDEX
    elif source_type in ("video", "demo"):
        path = data.get("path", "").strip()
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "sample.mp4")
        if not path or source_type == "demo":
            source = sample_path if os.path.exists(sample_path) else "inputs/sample.mp4"
        else:
            source = path
    else:
        return jsonify({"error": "Invalid source type", "code": "INVALID_SOURCE"}), 400

    # Reset
    session_id = start_session(source_type)
    state.update({
        "running"      : True,
        "source_type"  : source_type,
        "session_id"   : session_id,
        "frame_count"  : 0,
        "fps"          : 0.0,
        "total_persons": 0,
        "quadrants"    : {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0},
        "history"      : [],
    })
    stop_event.clear()
    threading.Thread(target=detection_loop, args=(source,), daemon=True).start()
    return jsonify({"success": True, "status": "started", "source": source_type, "session_id": session_id})


@app.route("/api/stream/stop", methods=["POST"])
@app.route("/api/stop", methods=["POST"])
def api_stop():
    stop_event.set()
    state["running"] = False
    return jsonify({"success": True, "status": "stopped"})


@app.route("/api/stream/status", methods=["GET"])
@app.route("/api/status", methods=["GET"])
def api_status():
    s = {k: v for k, v in state.items() if k != "history"}
    s["appliances"] = get_appliances_state(state.get("total_persons", 0))
    s["conf_threshold"] = round(float(detector.conf), 2)
    return jsonify(s)


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        data = request.json or {}
        if "conf_threshold" in data:
            try:
                val = float(data["conf_threshold"])
                detector.conf = max(0.10, min(0.90, round(val, 2)))
            except (TypeError, ValueError):
                return jsonify({"error": "Invalid confidence threshold"}), 400
    return jsonify({
        "success": True,
        "conf_threshold": round(float(detector.conf), 2)
    })


@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify(state["history"])


# ── Sample Datasets ───────────────────────────────────────────────────────────
@app.route("/api/samples", methods=["GET"])
@app.route("/api/sample_list", methods=["GET"])
def api_sample_list():
    """Return available sample images for instant 1-click testing."""
    samples = [
        {"id": "sample_1.jpg", "label": "Classroom View 1", "badge": "34 Students", "desc": "Mid-density lecture"},
        {"id": "sample_2.jpg", "label": "Classroom View 2", "badge": "56 Students", "desc": "Full capacity hall"},
        {"id": "sample_3.jpg", "label": "Classroom View 3", "badge": "12 Students", "desc": "Seminar grouping"}
    ]
    return jsonify(samples)


@app.route("/api/samples/<name>", methods=["GET", "POST"])
@app.route("/api/sample_image/<name>", methods=["GET", "POST"])
def api_sample_image(name):
    """Load and infer sample image directly without requiring file upload."""
    if not name.endswith(".jpg"):
        name = f"sample_{name}.jpg" if not name.startswith("sample_") else f"{name}.jpg"

    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", name)
    if not os.path.exists(img_path):
        return jsonify({"error": f"Sample dataset '{name}' not found"}), 404

    frame = cv2.imread(img_path)
    if frame is None:
        return jsonify({"error": "Cannot read sample image file"}), 400

    detections = detector.detect(frame)
    counts     = get_quadrant_counts(frame, detections)
    total      = sum(counts.values())

    annotated  = visualizer.draw(frame.copy(), detections)
    visualizer.draw_quadrants(annotated, counts)

    _, buf    = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64   = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    # Log to DB as a single-frame session
    sid = start_session(f"sample_{name}")
    record_frame(sid, 1, 0.0, counts["Q1"], counts["Q2"], counts["Q3"], counts["Q4"])
    flush_all()
    end_session(sid)

    state.update({
        "total_persons": total,
        "quadrants"    : counts,
        "source_type"  : "sample_image",
    })

    det_list = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        det_list.append({
            "id"        : i + 1,
            "label"     : d["label"],
            "confidence": round(d["confidence"] * 100, 1),
            "bbox"      : [int(x) for x in d["bbox"]],
            "quadrant"  : point_quadrant(frame, cx, cy),
        })

    return jsonify({
        "success"        : True,
        "sample_name"    : name,
        "image"          : img_b64,
        "quadrants"      : counts,
        "total"          : total,
        "appliances"     : get_appliances_state(total),
        "detection_count": len(detections),
        "detections"     : det_list,
    })


# ── Image Snapshot Inference ──────────────────────────────────────────────────
@app.route("/api/detect/image", methods=["POST"])
@app.route("/api/detect_image", methods=["POST"])
def api_detect_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    img_bytes = file.read()
    if not img_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Cannot decode image. Please provide a standard JPG, PNG, or WebP."}), 400

    # Normalize image dimensions if excessively large to ensure ultra-fast inference and clear HUD labels
    h, w = frame.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    detections = detector.detect(frame)
    counts     = get_quadrant_counts(frame, detections)
    total      = sum(counts.values())

    annotated  = visualizer.draw(frame.copy(), detections)
    visualizer.draw_quadrants(annotated, counts)

    _, buf    = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64   = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    # Log to DB as a single-frame session
    sid = start_session("custom_image")
    record_frame(sid, 1, 0.0, counts["Q1"], counts["Q2"], counts["Q3"], counts["Q4"])
    flush_all()
    end_session(sid)

    state.update({
        "total_persons": total,
        "quadrants"    : counts,
        "source_type"  : "custom_image",
    })

    det_list = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        det_list.append({
            "id"        : i + 1,
            "label"     : d["label"],
            "confidence": round(d["confidence"] * 100, 1),
            "bbox"      : [int(x) for x in d["bbox"]],
            "quadrant"  : point_quadrant(frame, cx, cy),
        })

    return jsonify({
        "success"        : True,
        "filename"       : file.filename,
        "image"          : img_b64,
        "quadrants"      : counts,
        "total"          : total,
        "appliances"     : get_appliances_state(total),
        "detection_count": len(detections),
        "detections"     : det_list,
    })


# ── Historical Analytics & Persistence ────────────────────────────────────────
@app.route("/api/analytics", methods=["GET"])
@app.route("/api/db/analytics", methods=["GET"])
def api_db_analytics():
    analytics = get_analytics()
    hourly = get_hourly_summary()

    if hourly:
        pcts = [get_appliances_state(round(h["avg_persons"]))["savings_pct"] for h in hourly]
        analytics["efficiency_index"] = round(sum(pcts) / len(pcts), 1)
    else:
        analytics["efficiency_index"] = get_appliances_state(0)["savings_pct"]

    return jsonify({
        "success"  : True,
        "analytics": analytics,
        "hourly"   : hourly
    })


@app.route("/api/sessions", methods=["GET"])
@app.route("/api/db/sessions", methods=["GET"])
def api_db_sessions():
    limit = int(request.args.get("limit", 12))
    return jsonify(get_sessions(limit))


@app.route("/api/sessions/<int:session_id>/timeline", methods=["GET"])
def api_session_timeline(session_id):
    frames = get_session_timeline(session_id)
    return jsonify({
        "success": True,
        "session_id": session_id,
        "frames": frames
    })


@app.route("/api/export/csv", methods=["GET"])
@app.route("/api/db/export_csv", methods=["GET"])
def api_db_export_csv():
    data = get_hourly_summary()
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["Date/Time", "Average Persons", "Peak Persons", "Total Frames Processed"])
    for r in data:
        dt = datetime.datetime.fromtimestamp(r["hour_ts"]).strftime('%Y-%m-%d %H:%M:%S')
        cw.writerow([dt, r["avg_persons"], r["peak_persons"], r["frame_count"]])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=classroom_hourly_stats.csv"
    output.headers["Content-type"] = "text/csv"
    return output


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 52)
    print("  🎓  Smart Classroom AI Dashboard")
    print("  ➜  Open: http://localhost:5050")
    print("═" * 52 + "\n")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
