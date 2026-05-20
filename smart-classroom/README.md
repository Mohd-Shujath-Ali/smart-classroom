# Smart Classroom AI Dashboard 🎓

![Smart Classroom AI](/smart-classroom/assets/placeholder.png) 

A real-time Computer Vision system and web dashboard designed for modern educational environments. It uses machine learning to actively monitor classroom occupancy, automatically calculates energy-saving appliance states (fans & lights) based on student density, and stores historical analytics data in a local SQLite database for auditing.

---

## 🌟 Key Features

### 1. Real-Time Occupancy & Quadrant Analysis
- Powered by `Ultralytics YOLOv8`, detecting persons in real time.
- Mathematically divides room camera feeds into four distinct zones (Top-Left, Top-Right, Bottom-Left, Bottom-Right) to visualize student distribution.
- Supports live webcam, local `.mp4` video files, or individual `.jpg`/`.png` image uploads.

### 2. Intelligent Energy Management (Automated Appliances)
- Links live occupant count directly to physical room appliance rules:
  - **0 Students:** 0 Fans, 0 Lights _(Energy Saving Standby)_
  - **1–10 Students:** 2 Fans, 4 Lights _(Low Power Mode)_
  - **11–19 Students:** 4 Fans, 6 Lights _(Medium Power Mode)_
  - **20+ Students:** 6 Fans, 8 Lights _(Maximum Cooling Mode)_
- The web dashboard dynamically reflects fan and light states in real time.

### 3. Persistent Analytics Backend (SQLite + CSV Export)
- Employs a thread-safe SQLite memory buffer using Write-Ahead Logging (WAL) to record metrics without lagging the video feed.
- Tracks detection sessions with timestamps, peak human counts, average persons, and total frames processed.
- **Data Export:** Export rolling 24-hour analytics to a spreadsheet via the **"Download CSV"** button, or export the tracking chart using the **"Download Graph"** button.

### 4. Interactive Web Dashboard
- Built with **Flask**, **Flask-SocketIO**, and raw modern CSS styling.
- WebSocket connections push sub-second detection metrics to the frontend alongside the MJPEG video stream.

---

## 🛠️ Technology Stack
* **Computer Vision:** OpenCV (`cv2`), Ultralytics YOLOv8 (`models/best.pt`)
* **Backend Server:** Python 3, Flask, Flask-SocketIO
* **Database:** SQLite (`sqlite3`) with WAL mode
* **Frontend UI:** HTML5, Vanilla CSS, Vanilla JavaScript, Chart.js

---

## 🚀 Installation & Getting Started

**1. Clone the Repository:**
```bash
git clone https://github.com/Mohd-Shujath-Ali/smart-classroom.git
cd smart-classroom/smart-classroom
```

**2. Install Dependencies:**
Ensure you have Python 3 installed, then run:
```bash
pip install -r requirements.txt
```

**3. Run the Smart Dashboard Server:**

On Windows:
```powershell
python app.py
```

On macOS / Linux:
```bash
chmod +x run.sh
./run.sh
# or manually:
python app.py or python3 app.py
```

**4. Open the Interface:**
Once you see `* Running on http://127.0.0.1:5050`, navigate to:
[http://localhost:5050](http://localhost:5050)

**5. Standalone CLI Mode (Optional):**
You can also run detections directly in an OpenCV window without the web server:
```bash
python main.py --source webcam
python main.py --source inputs/sample_1.jpg
python main.py --source inputs/sample.mp4
```

---

## 📁 Directory Structure

```text
smart-classroom/
│
├── app.py                # Main Flask/Socket.IO backend server
├── main.py               # Standalone CLI / OpenCV runner
├── config.py             # Global configurations (Model paths, threshold limits)
├── run.sh                # Helper script to execute the app environment
├── requirements.txt      # Python package dependencies
├── Final_Project_Report.md # Project documentation and report
│
├── assets/               # Dashboard screenshots and visuals
│   ├── placeholder.jpg
│   ├── placeholder.png
│   └── analysis.png
│
├── db/                   # Database Layer
│   ├── database.py       # SQL Queries, memory buffers, & session logic
│   └── classroom.db      # Automatically generated SQLite storage file 
│
├── detector/             # Inference logic
│   └── yolo_detector.py  # Wrapper module for running the YOLO model
│
├── models/               # Model weights
│   ├── best.pt           # The trained YOLOv8 PyTorch model
│   └── README.txt
│
├── inputs/               # Test media
│   ├── sample_1.jpg
│   ├── sample_2.jpg
│   └── sample_3.jpg
│
├── templates/            # Frontend Web UI
│   └── index.html        # The entire dashboard interface & Chart.js logic
│
└── utils/
    └── visualizer.py     # Drawing logic for bounding boxes, quadrant UI, and text
```

---

## 💡 How to Use

1. **Dashboard Controls (Top bar):** Select your source: `Webcam`, `Video File`, or `Image`.
2. **Video Playback:** If you selected a Video File, enter the path (e.g. `inputs/sample.mp4`) and click **Start**.
3. **Automated Tracking:** As the model runs, quadrant counts, total occupant count, and Automated Appliances update in real time.
4. **Image Upload:** Switch to Image mode to drag and drop or upload a photo for instant detection and appliance evaluation.
5. **Export Findings:** In the **Historical Database Data** card, view 24-hour trends, click **CSV** to download a spreadsheet, or click **Graph** to download a PNG of the chart.

---

> _**Project Status:** Active Development & Maintained._
