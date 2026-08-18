"""
I2DMS - Intelligent In-Cabin Driver Monitoring System
Live Webcam Demo with YOLO Object Detection, MediaPipe Hand-to-Head Backup, & Speed Suppression

Run:
    python app.py
"""

from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from nicegui import ui
from ultralytics import YOLO
import os
import base64
import sqlite3
import threading
from datetime import datetime
from typing import List
import time

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import mediapipe as mp
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing

from nicegui import app, ui
from ultralytics import YOLO

# CONFIG & INITIALIZATION

APP_NAME = "I2DMS - Driver Monitoring System (Webcam)"
DB_FILE = "i2dms.db"
MIN_SPEED_VIOLATION = 10.0  # km/h threshold

# Load YOLO model
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

# Initialize MediaPipe Hands
print("Loading MediaPipe Hands...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# DATABASE MANAGER

class DatabaseManager:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS violations(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    speed REAL,
                    confidence REAL,
                    reason TEXT,
                    event_type TEXT DEFAULT 'VIOLATION' 
                )
                """)
            conn.commit()

    def log_violation(self, speed: float, confidence: float, reason: str, event_type: str = "VIOLATION"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO violations (timestamp, speed, confidence, reason, event_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, speed, confidence, reason, event_type)
            )
            conn.commit()

db = DatabaseManager()

# STATE MANAGER

class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self.speed: float = 0.0
        self.phone_detected: bool = False
        self.phone_confidence: float = 0.0
        self.detection_source: str = "None"
        self.driver_status: str = "SAFE"
        self.decision: str = "Camera active. Looking for driver behavior..."
        self.risk: float = 0.0
        self.timeline: List[dict] = []
        self.latest_frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
        self.is_running: bool = True
        self.last_db_log_time: float = 0.0
        
    def set_frame(self, frame: np.ndarray):
        with self._lock:
            self.latest_frame = frame.copy()

    def get_frame_base64(self) -> str:
        with self._lock:
            _, buffer = cv2.imencode('.jpg', self.latest_frame)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_str}"

    def add_event(self, text: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            if not self.timeline or self.timeline[0]["text"] != text:
                self.timeline.insert(0, {"time": timestamp, "text": text, "level": level})
                if len(self.timeline) > 10:
                    self.timeline.pop()

state = AppState()

# WEBCAM PROCESSING LOOP

def process_webcam():
    """Background thread for continuous webcam capture, YOLO + MediaPipe processing."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while state.is_running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip horizontally for natural webcam view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 1. YOLO INFERENCE (Lower confidence threshold to 0.15 for PoC) 
        yolo_results = yolo_model(frame, conf=0.15, imgsz=320, verbose=False)
        result = yolo_results[0]
        annotated_frame = result.plot()

        yolo_phone_found = False
        yolo_phone_conf = 0.0

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            conf = float(box.conf[0])

            if cls_id == 67 or cls_name in ["cell phone", "phone"]:
                yolo_phone_found = True
                yolo_phone_conf = max(yolo_phone_conf, conf)

        # 2. MEDIAPIPE HAND FALLBACK (Head/Ear Level Detection)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)

        hand_near_head = False
        hand_conf = 0.0

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand skeleton overlay
                mp_drawing.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )

                # Wrist (Index 0) and Index Finger Tip (Index 8) normalized Y positions
                wrist_y = hand_landmarks.landmark[0].y
                index_tip_y = hand_landmarks.landmark[8].y

                # If wrist or index tip is in the upper half of the screen (near head/ear)
                if wrist_y < 0.55 or index_tip_y < 0.50:
                    hand_near_head = True
                    hand_conf = 0.85  # Fallback rule confidence score
                    break

        # Combine Detection Logic
        phone_found = yolo_phone_found or hand_near_head
        
        if yolo_phone_found:
            source = "YOLO (Object)"
            final_conf = yolo_phone_conf
        elif hand_near_head:
            source = "MediaPipe (Gesture)"
            final_conf = hand_conf
        else:
            source = "Clear"
            final_conf = 0.0

        state.phone_detected = phone_found
        state.phone_confidence = final_conf
        state.detection_source = source
        state.set_frame(annotated_frame)

        # 3. BUSINESS LOGIC & SPEED SUPPRESSION 
        current_speed = state.speed
        now = time.time()  # Track current timestamp

        if phone_found:
            if current_speed > MIN_SPEED_VIOLATION:
                state.driver_status = "VIOLATION"
                state.decision = f"Distraction Breach! [{source}] @ {current_speed:.0f} km/h."
                state.risk = min(98.0, 60.0 + current_speed * 0.3)
                state.add_event(f"VIOLATION [{source}] @ {current_speed:.0f} km/h", "DANGER")
                
                # Write to DB at most once every 2 seconds
                if now - state.last_db_log_time > 2.0:
                    db.log_violation(current_speed, final_conf, f"Phone distraction ({source})", "VIOLATION")
                    state.last_db_log_time = now
            else:
                state.driver_status = "SAFE"
                state.decision = f"Phone/Gesture detected, suppressed (Speed = {current_speed:.0f} km/h ≤ 10)."
                state.risk = 15.0
                state.add_event(f"Suppressed (Stationary) @ {current_speed:.0f} km/h", "INFO")
                
                # Write to DB at most once every 2 seconds
                if now - state.last_db_log_time > 2.0:
                    db.log_violation(current_speed, final_conf, f"Suppressed distraction ({source})", event_type="INFO")
                    state.last_db_log_time = now
        else:
            state.driver_status = "SAFE"
            state.decision = "Driver attentive. No phone detected."
            state.risk = max(2.0, current_speed * 0.08)

    cap.release()

# Start background camera thread
threading.Thread(target=process_webcam, daemon=True).start()

# DASHBOARD PAGE

@ui.page('/')
def main_page():
    ui.dark_mode().enable()
    ui.query("body").style("background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif;")

    # Header
    with ui.header().classes("bg-slate-900 border-b border-slate-800 items-center justify-between px-6 py-3"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("videocam", color="primary").classes("text-3xl")
            ui.label(APP_NAME).classes("text-xl font-bold tracking-wide")
        ui.badge("HYBRID YOLO + MEDIAPIPE", color="green").props("outline")

    # Main Content Grid
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):

        # CONTROL BAR: SPEED SLIDER
        with ui.card().classes("w-full bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl"):
            with ui.row().classes("w-full items-center justify-between gap-6 flex-wrap"):
                with ui.column().classes("gap-1 flex-1 min-w-[280px]"):
                    with ui.row().classes("justify-between text-xs text-slate-400 w-full"):
                        ui.label("Simulated Vehicle Speed")
                        speed_display = ui.label("0 km/h").classes("text-emerald-400 font-bold")
                    
                    def on_speed_change(e):
                        state.speed = float(e.value)
                        speed_display.set_text(f"{e.value:.0f} km/h")

                    ui.slider(min=0, max=120, value=0, step=1, on_change=on_speed_change).classes("w-full")

                ui.label("Adjust speed slider").classes("text-xs text-slate-500 max-w-xs")

        # MAIN DISPLAY
        with ui.row().classes("w-full grid grid-cols-1 lg:grid-cols-3 gap-6 items-start"):
            
            # 1. Live Camera Stream
            with ui.card().classes("lg:col-span-2 bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl"):
                ui.label("In-Cabin Driver Feed").classes("text-sm font-semibold uppercase tracking-wider text-slate-400 mb-2")
                video_image = ui.interactive_image(state.get_frame_base64()).classes("w-full rounded-xl overflow-hidden aspect-video object-cover")

            # 2. Live Telemetry Panel
            with ui.card().classes("bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl flex flex-col gap-4"):
                ui.label("Real-Time Telemetry").classes("text-sm font-semibold uppercase tracking-wider text-slate-400")
                
                status_chip = ui.label("SAFE").classes("text-center font-bold text-2xl py-3 rounded-xl transition-all duration-300")
                
                ui.separator().classes("bg-slate-800")
                
                with ui.column().classes("gap-3 text-sm w-full"):
                    with ui.row().classes("justify-between items-center w-full"):
                        ui.label("Current Speed").classes("text-slate-400")
                        speed_label = ui.label("0 km/h").classes("font-semibold text-base")

                    with ui.row().classes("justify-between items-center w-full"):
                        ui.label("Phone / Distraction").classes("text-slate-400")
                        phone_label = ui.label("Not Detected").classes("font-semibold text-base")

                    with ui.row().classes("justify-between items-center w-full"):
                        ui.label("Detection Source").classes("text-slate-400")
                        source_label = ui.label("None").classes("font-semibold text-base text-sky-400")

                    with ui.row().classes("justify-between items-center w-full"):
                        ui.label("Confidence").classes("text-slate-400")
                        confidence_label = ui.label("0%").classes("font-semibold text-base")

                ui.separator().classes("bg-slate-800")

                with ui.column().classes("w-full gap-1"):
                    with ui.row().classes("justify-between text-xs text-slate-400 w-full"):
                        ui.label("Distraction Risk Index")
                        risk_value_label = ui.label("0%")
                    risk_bar = ui.linear_progress(value=0.0, show_value=False).classes("h-2 rounded-full")

                with ui.column().classes("w-full bg-slate-800/50 p-3 rounded-lg border border-slate-800"):
                    ui.label("System Decision").classes("text-xs text-slate-400 font-medium")
                    decision_label = ui.label("Monitoring...").classes("text-sm text-slate-200 mt-1")

        # 3. Timeline Audit
        with ui.card().classes("w-full bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl"):
            ui.label("Live Violation Log").classes("text-sm font-semibold uppercase tracking-wider text-slate-400 mb-3")
            timeline_container = ui.column().classes("w-full gap-2")

    # REFRESH FUNCTION INSIDE PAGE CONTEXT
    def refresh_ui():
        video_image.set_source(state.get_frame_base64())
        speed_label.set_text(f"{state.speed:.0f} km/h")
        phone_label.set_text("DETECTED" if state.phone_detected else "Clear")
        source_label.set_text(state.detection_source)
        confidence_label.set_text(f"{state.phone_confidence:.0%}")
        decision_label.set_text(state.decision)
        
        risk_pct = state.risk / 100.0
        risk_bar.set_value(risk_pct)
        risk_value_label.set_text(f"{state.risk:.0f}%")

        if state.driver_status == "SAFE":
            status_chip.set_text("SAFE")
            status_chip.style("background-color: #10b98120; color: #10b981; border: 1px solid #10b981;")
        else:
            status_chip.set_text("VIOLATION BREACH")
            status_chip.style("background-color: #ef444420; color: #ef4444; border: 1px solid #ef4444;")

        timeline_container.clear()
        with timeline_container:
            if not state.timeline:
                ui.label("No violations recorded yet.").classes("text-slate-500 text-sm")
            for ev in state.timeline[:5]:
                with ui.row().classes("w-full items-center justify-between p-2 rounded-lg bg-slate-800/40 text-sm border border-slate-800"):
                    with ui.row().classes("items-center gap-3"):
                        ui.label(ev["time"]).classes("text-slate-400 text-xs font-mono")
                        ui.label(ev["text"]).classes("text-slate-200")
                    if ev["level"] == "DANGER":
                        ui.badge("VIOLATION", color="red").props("dense")
                    else:
                        ui.badge("INFO", color="blue").props("dense")

    ui.timer(0.1, refresh_ui)


ui.run(title="I2DMS Hybrid Monitor", reload=False, port=8080)
