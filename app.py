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
import requests
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for Vercel frontend

# Create separate queues for each video feed
frame_queues = [queue.Queue(maxsize=10) for _ in range(4)]
data_queues = [queue.Queue(maxsize=1) for _ in range(4)]

# Sample public video URLs (free to use)
SAMPLE_VIDEOS = [
    "https://sample-videos.com/zip/10/mp4/SampleVideo_720x480_1mb.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4",
    "https://file-examples.com/storage/fe86d2733df0b6392313cb4/2017/10/file_example_MP4_480_1_5MG.mp4"
]

class VehicleDetector:
    def __init__(self):
        try:
            # Try to load YOLOv8 nano model (smaller, faster)
            self.model = YOLO("yolov8n.pt")
        except:
            # Fallback - download if not available
            print("Downloading YOLOv8 nano model...")
            self.model = YOLO("yolov8n.pt")
            
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

        # Detection lines
        self.limit_lines = None

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
            if elapsed_time >= 5:
                self.signal_state = "GREEN"
                self.signal_start_time = current_time
                self.vehicles_passed = 0
                self.current_green_time = self.calculate_green_time(len(self.total_count))

    def process_frame(self, frame):
        if frame is None:
            return self.create_dummy_frame(), 0, 30, "GREEN"
            
        if self.limit_lines is None:
            self.initialize_lines(frame)

        try:
            results = self.model(frame, stream=True, verbose=False)
            detections = np.empty((0, 5))

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        
                        if cls < len(self.class_names):
                            current_class = self.class_names[cls]
                            if current_class in self.target_classes and conf > 0.3:
                                current_array = np.array([x1, y1, x2, y2, conf])
                                detections = np.vstack((detections, current_array))

            tracked_objects = self.tracker.update(detections)

            # Draw detection lines
            for limit in self.limit_lines:
                cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]), (250, 182, 122), 2)

            # Process tracked objects
            for result in tracked_objects:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(111, 237, 235))
                cvzone.putTextRect(frame, f'#{int(id)}', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1, offset=5,
                                   colorR=(56, 245, 213), colorT=(25, 26, 25))

                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(frame, (cx, cy), 5, (22, 192, 240), cv2.FILLED)

                # Check if vehicle crossed the line
                for limit in self.limit_lines:
                    if (limit[0] < cx < limit[2] and
                            limit[1] - 15 < cy < limit[1] + 15 and
                            id not in self.total_count):
                        self.total_count.append(id)
                        cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]), (12, 202, 245), 3)
                        if self.signal_state == "GREEN":
                            self.vehicles_passed += 1

        except Exception as e:
            print(f"Detection error: {e}")

        self.update_signal_state()
        self.draw_signal_info(frame)

        return frame, len(self.total_count), self.current_green_time, self.signal_state

    def draw_signal_info(self, frame):
        # Signal state display
        signal_color = (0, 255, 0) if self.signal_state == "GREEN" else (0, 255, 255) if self.signal_state == "YELLOW" else (0, 0, 255)
        cv2.rectangle(frame, (20, 20), (250, 120), signal_color, -1)
        
        cvzone.putTextRect(frame, f'Signal: {self.signal_state}', (30, 45),
                           scale=1, thickness=2, offset=3, colorR=signal_color, colorT=(0, 0, 0))
        cvzone.putTextRect(frame, f'Count: {len(self.total_count)}', (30, 75),
                           scale=1, thickness=2, offset=3, colorR=signal_color, colorT=(0, 0, 0))
        cvzone.putTextRect(frame, f'Time: {self.current_green_time}s', (30, 105),
                           scale=1, thickness=2, offset=3, colorR=signal_color, colorT=(0, 0, 0))

    def create_dummy_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Traffic Monitor", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

# Create detector instances
detectors = [VehicleDetector() for _ in range(4)]

# Video processing function
def video_processing_thread(feed_id):
    video_url = SAMPLE_VIDEOS[feed_id % len(SAMPLE_VIDEOS)]
    
    while True:
        cap = None
        try:
            # Try to open video stream
            cap = cv2.VideoCapture(video_url)
            
            # If video fails, create dummy frames
            if not cap.isOpened():
                raise Exception("Cannot open video")
                
            # Set buffer size for better performance
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS for better performance
            
            frame_count = 0
            
            while True:
                success, frame = cap.read()
                
                if not success:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Process every 2nd frame for performance
                frame_count += 1
                if frame_count % 2 != 0:
                    continue

                # Resize frame for better performance
                frame = cv2.resize(frame, (640, 480))
                
                # Process frame
                processed_frame, count, green_time, signal_state = detectors[feed_id].process_frame(frame)

                # Update data queue
                data = {
                    "count": count,
                    "green_time": green_time,
                    "signal_state": signal_state,
                    "feed_id": feed_id
                }

                try:
                    data_queues[feed_id].put(data, block=False)
                except queue.Full:
                    try:
                        data_queues[feed_id].get_nowait()
                        data_queues[feed_id].put(data, block=False)
                    except queue.Empty:
                        pass

                # Encode frame
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = buffer.tobytes()

                # Update frame queue
                try:
                    frame_queues[feed_id].put(frame_bytes, block=False)
                except queue.Full:
                    try:
                        frame_queues[feed_id].get_nowait()
                        frame_queues[feed_id].put(frame_bytes, block=False)
                    except queue.Empty:
                        pass
                        
                time.sleep(0.05)  # Small delay for performance

        except Exception as e:
            print(f"Video processing error for feed {feed_id}: {e}")
            
            # Create dummy frame on error
            dummy_frame = detectors[feed_id].create_dummy_frame()
            cv2.putText(dummy_frame, f"Feed {feed_id+1} - Connection Error", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', dummy_frame)
            frame_bytes = buffer.tobytes()
            
            try:
                frame_queues[feed_id].put(frame_bytes, block=False)
            except queue.Full:
                pass
            
            time.sleep(2)  # Wait before retry
            
        finally:
            if cap:
                cap.release()

# Routes
@app.route('/')
def index():
    return jsonify({
        "message": "Traffic Monitor Backend API", 
        "status": "running",
        "endpoints": ["/health", "/video_feed/<feed_id>", "/get_data/<feed_id>"]
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "feeds": 4
    })

def generate_frames(feed_id):
    while True:
        try:
            frame_bytes = frame_queues[feed_id].get(timeout=5)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            # Send dummy frame on timeout
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy, f"Feed {feed_id+1} Loading...", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', dummy)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed/<int:feed_id>')
def video_feed(feed_id):
    if 0 <= feed_id < 4:
        return Response(generate_frames(feed_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return jsonify({"error": "Invalid feed ID"}), 404

@app.route('/get_data/<int:feed_id>')
def get_data(feed_id):
    if 0 <= feed_id < 4:
        try:
            data = data_queues[feed_id].get_nowait()
            return jsonify(data)
        except queue.Empty:
            return jsonify({
                "count": 0, 
                "green_time": 30, 
                "signal_state": "GREEN",
                "feed_id": feed_id
            })
    return jsonify({"error": "Invalid feed ID"}), 404

@app.route('/all_data')
def get_all_data():
    all_data = {}
    for i in range(4):
        try:
            data = data_queues[i].get_nowait()
            data_queues[i].put(data)  # Put it back
            all_data[f"feed_{i}"] = data
        except queue.Empty:
            all_data[f"feed_{i}"] = {
                "count": 0,
                "green_time": 30,
                "signal_state": "GREEN",
                "feed_id": i
            }
    return jsonify(all_data)

if __name__ == '__main__':
    # Start video processing threads
    for i in range(4):
        thread = threading.Thread(target=video_processing_thread, args=(i,), daemon=True)
        thread.start()

    # Get port from environment (Render requirement)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)