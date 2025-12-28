from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import *
import threading
import queue
import time
import os
import tempfile
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Import cloud video handlers
from cloud_video_handler import CloudVideoHandler, PublicVideoHandler, LiveStreamHandler

# Create separate queues for each video feed
frame_queues = [queue.Queue(maxsize=10) for _ in range(4)]
data_queues = [queue.Queue(maxsize=1) for _ in range(4)]

class VehicleDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # Use smaller model for faster processing
        self.tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.1)
        self.total_count = []

        # Constants
        self.MAX_SIGNAL_TIME = 120
        self.MIN_SIGNAL_TIME = 30
        self.MAX_TRAFFIC = 50
        self.YELLOW_TIME = 5

        # Class names for detection
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]
        self.target_classes = ["car", "truck", "bus", "motorbike"]

        # Detection lines (will be initialized with frame dimensions)
        self.limit_lines = None
        self.vehicle_counts_history = []

        # Traffic signal state
        self.signal_state = "GREEN"
        self.signal_start_time = time.time()
        self.current_green_time = 30
        self.vehicles_passed = 0

    def initialize_lines(self, frame):
        height, width = frame.shape[:2]
        y1 = height * 0.6
        y2 = y1 + 20

        self.limit_lines = [
            [int(width * 0.2), int(y1), int(width * 0.8), int(y1)],
            [int(width * 0.2), int(y2), int(width * 0.8), int(y2)]
        ]

    def calculate_green_time(self, vehicle_count):
        density_factor = self.MAX_SIGNAL_TIME / self.MAX_TRAFFIC
        green_time = max(self.MIN_SIGNAL_TIME, min(density_factor * vehicle_count, self.MAX_SIGNAL_TIME))
        return int(green_time)

    def update_signal_state(self):
        current_time = time.time()
        elapsed_time = current_time - self.signal_start_time

        if self.signal_state == "GREEN":
            if self.vehicles_passed >= 2 and elapsed_time >= self.current_green_time:
                self.signal_state = "YELLOW"
                self.signal_start_time = current_time
        elif self.signal_state == "YELLOW":
            if elapsed_time >= self.YELLOW_TIME:
                self.signal_state = "RED"
                self.signal_start_time = current_time
        elif self.signal_state == "RED":
            if elapsed_time >= 5:  # Red light duration
                self.signal_state = "GREEN"
                self.signal_start_time = current_time
                self.vehicles_passed = 0
                self.current_green_time = self.calculate_green_time(len(self.total_count))

    def process_frame(self, frame):
        if self.limit_lines is None:
            self.initialize_lines(frame)

        results = self.model(frame, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                if current_class in self.target_classes and conf > 0.3:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        tracked_objects = self.tracker.update(detections)

        for limit in self.limit_lines:
            cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]),
                     (250, 182, 122), 2)

        for result in tracked_objects:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2,
                              colorR=(111, 237, 235))
            cvzone.putTextRect(frame, f'#{int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=1, offset=10,
                               colorR=(56, 245, 213), colorT=(25, 26, 25))

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(frame, (cx, cy), 5, (22, 192, 240), cv2.FILLED)

            for limit in self.limit_lines:
                if (limit[0] < cx < limit[2] and
                        limit[1] - 15 < cy < limit[1] + 15 and
                        id not in self.total_count):
                    self.total_count.append(id)
                    cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]),
                             (12, 202, 245), 3)
                    if self.signal_state == "GREEN":
                        self.vehicles_passed += 1

        self.update_signal_state()

        # Display signal state and other information
        signal_color = (0, 255, 0) if self.signal_state == "GREEN" else (0, 255, 255) if self.signal_state == "YELLOW" else (0, 0, 255)
        cv2.rectangle(frame, (20, 20), (200, 100), signal_color, -1)
        cvzone.putTextRect(frame, f'Signal: {self.signal_state}', (30, 40),
                           scale=1, thickness=2, offset=5,
                           colorR=signal_color, colorT=(0, 0, 0))
        cvzone.putTextRect(frame, f'Count: {len(self.total_count)}', (30, 70),
                           scale=1, thickness=2, offset=5,
                           colorR=signal_color, colorT=(0, 0, 0))
        
        if self.signal_state == "GREEN":
            cvzone.putTextRect(frame, f'Green Time: {self.current_green_time}s', (30, 100),
                               scale=1, thickness=2, offset=5,
                               colorR=signal_color, colorT=(0, 0, 0))

        return frame, len(self.total_count), self.current_green_time, self.signal_state

# Create detector instances for each feed
detectors = [VehicleDetector() for _ in range(4)]

# Video source handlers
cloud_handler = CloudVideoHandler()
public_handler = PublicVideoHandler()
live_handler = LiveStreamHandler()

# Global variables for video sources
current_video_sources = [None, None, None, None]
video_source_type = "demo"  # "demo", "cloud", "live", "upload"

def video_processing_thread(feed_id):
    global current_video_sources
    
    while True:
        cap = None
        
        # Initialize video source based on type
        if video_source_type == "demo":
            # Use sample/demo videos
            cap = public_handler.get_sample_video_stream(feed_id % len(public_handler.sample_videos))
        elif video_source_type == "cloud" and current_video_sources[feed_id]:
            # Use cloud storage videos
            cap = cloud_handler.get_video_stream_from_s3(current_video_sources[feed_id])
        elif video_source_type == "live":
            # Use live camera feeds
            cap = live_handler.get_live_stream(feed_id)
        elif video_source_type == "upload" and current_video_sources[feed_id]:
            # Use uploaded videos
            cap = cv2.VideoCapture(current_video_sources[feed_id])
        
        if cap is None:
            # Fallback to webcam or dummy frame
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                # Create dummy frame if no video source available
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy_frame, f"Feed {feed_id+1} - No Source", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', dummy_frame)
                frame_bytes = buffer.tobytes()
                
                try:
                    frame_queues[feed_id].put(frame_bytes, block=False)
                except queue.Full:
                    pass
                
                time.sleep(0.1)
                continue

        success, frame = cap.read()
        if not success:
            if hasattr(cap, 'set'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        processed_frame, count, green_time, signal_state = detectors[feed_id].process_frame(frame)

        data = {
            "count": count,
            "green_time": green_time,
            "signal_state": signal_state
        }

        try:
            data_queues[feed_id].put(data, block=False)
        except queue.Full:
            try:
                data_queues[feed_id].get_nowait()
                data_queues[feed_id].put(data, block=False)
            except queue.Empty:
                pass

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        try:
            frame_queues[feed_id].put(frame_bytes, block=False)
        except queue.Full:
            try:
                frame_queues[feed_id].get_nowait()
                frame_queues[feed_id].put(frame_bytes, block=False)
            except queue.Empty:
                pass

@app.route('/')
def index():
    return jsonify({"message": "Traffic Monitor Backend API", "status": "running"})

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})

def generate_frames(feed_id):
    while True:
        frame_bytes = frame_queues[feed_id].get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/<int:feed_id>')
def video_feed(feed_id):
    if 0 <= feed_id < 4:
        return Response(generate_frames(feed_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Invalid feed ID", 404

@app.route('/get_data/<int:feed_id>')
def get_data(feed_id):
    if 0 <= feed_id < 4:
        try:
            data = data_queues[feed_id].get_nowait()
            return jsonify(data)
        except queue.Empty:
            return jsonify({"count": 0, "green_time": 30, "signal_state": "GREEN"})
    return jsonify({"error": "Invalid feed ID"}), 404

@app.route('/set_video_source', methods=['POST'])
def set_video_source():
    global video_source_type, current_video_sources
    
    data = request.get_json()
    source_type = data.get('type', 'demo')
    sources = data.get('sources', [])
    
    video_source_type = source_type
    current_video_sources = sources + [None] * (4 - len(sources))
    
    return jsonify({"message": f"Video source set to {source_type}", "sources": current_video_sources})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    feed_id = request.form.get('feed_id', 0)
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        # Save to temporary directory (for demo purposes)
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        
        # Update video source
        global current_video_sources, video_source_type
        video_source_type = "upload"
        current_video_sources[int(feed_id)] = file_path
        
        return jsonify({"message": "Video uploaded successfully", "file_path": file_path})

if __name__ == '__main__':
    # Start video processing threads for each feed
    for i in range(4):
        threading.Thread(target=video_processing_thread, args=(i,), daemon=True).start()

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)