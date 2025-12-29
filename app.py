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
from intelligent_traffic_optimizer import IntelligentTrafficOptimizer, VehicleData, LaneMetrics, SignalPhase

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Import cloud video handlers
from cloud_video_handler import CloudVideoHandler, PublicVideoHandler, LiveStreamHandler

# Create separate queues for each video feed
frame_queues = [queue.Queue(maxsize=10) for _ in range(4)]
data_queues = [queue.Queue(maxsize=1) for _ in range(4)]

# Global optimizer instance (shared across all detectors for phase management)
global_optimizer = None

class VehicleDetector:
    def __init__(self):
        try:
            print("Loading YOLO model...")
            self.model = YOLO("yolov8n.pt")  # Use smaller model for faster processing
            self.tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.1)
            self.total_count = []
            
            # Initialize intelligent optimizer (will be overridden by global optimizer)
            self.optimizer = IntelligentTrafficOptimizer()
            print("VehicleDetector initialized successfully")
        except Exception as e:
            print(f"Error initializing VehicleDetector: {e}")
            # Create minimal fallback
            self.model = None
            self.tracker = None
            self.total_count = []
            self.optimizer = None
        
        self.vehicles_data = []  # Store enhanced vehicle data
        self.bottleneck_strategies = {}
        self.lane_id = 0  # Will be set per detector instance
        
        # Traffic engineering standards
        self.MAX_SIGNAL_TIME = 120
        self.MIN_SIGNAL_TIME = 7
        self.YELLOW_TIME = 3
        self.ALL_RED_TIME = 2
        
        # Real-world vehicle classification
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]
        self.target_classes = ["car", "truck", "bus", "motorbike", "bicycle"]
        
        # Vehicle type mapping for optimization
        self.vehicle_type_map = {
            "car": "car",
            "truck": "truck", 
            "bus": "bus",
            "motorbike": "bike",
            "bicycle": "bike"
        }

        # Detection lines (will be initialized with frame dimensions)
        self.limit_lines = None
        self.vehicle_counts_history = []
        
        # Performance tracking
        self.last_optimization_time = time.time()
        self.optimization_interval = 5  # Optimize every 5 seconds

    def initialize_lines(self, frame):
        height, width = frame.shape[:2]
        y1 = height * 0.6
        y2 = y1 + 20

        self.limit_lines = [
            [int(width * 0.2), int(y1), int(width * 0.8), int(y1)],
            [int(width * 0.2), int(y2), int(width * 0.8), int(y2)]
        ]

    def get_signal_state(self):
        """Get signal state from global optimizer based on phase logic"""
        global global_optimizer
        if global_optimizer is None:
            return "GREEN"  # Fallback if optimizer not available
        
        return global_optimizer.get_signal_state(self.lane_id)
    
    def get_green_time(self):
        """Get remaining green time from global optimizer"""
        global global_optimizer
        if global_optimizer is None:
            return 30
        
        return global_optimizer.get_green_time_for_lane(self.lane_id)

    def calculate_lane_metrics(self, vehicle_count, vehicle_types=None):
        """Calculate lane metrics for this detector's lane"""
        
        if not vehicle_types:
            vehicle_types = ['car'] * vehicle_count
        
        # Create vehicle data for optimization
        vehicles = []
        for i, v_type in enumerate(vehicle_types[:vehicle_count]):
            vehicle = VehicleData(
                vehicle_id=i,
                vehicle_type=self.vehicle_type_map.get(v_type, 'car'),
                position=(0, 0),
                speed=0,
                queue_position=i,
                wait_time=0,
                priority_level=2 if v_type == 'emergency' else 0
            )
            vehicles.append(vehicle)
        
        # Analyze lane conditions using the optimizer
        lane_metrics = self.optimizer.analyze_lane_conditions(vehicles)
        
        return lane_metrics

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

        # Get signal state from global phase-based optimizer
        signal_state = self.get_signal_state()
        green_time = self.get_green_time()

        # Display signal state and other information
        signal_color = (0, 255, 0) if signal_state == "GREEN" else (0, 255, 255) if signal_state == "YELLOW" else (0, 0, 255)
        cv2.rectangle(frame, (20, 20), (200, 100), signal_color, -1)
        cvzone.putTextRect(frame, f'Signal: {signal_state}', (30, 40),
                           scale=1, thickness=2, offset=5,
                           colorR=signal_color, colorT=(0, 0, 0))
        cvzone.putTextRect(frame, f'Count: {len(self.total_count)}', (30, 70),
                           scale=1, thickness=2, offset=5,
                           colorR=signal_color, colorT=(0, 0, 0))
        
        if signal_state == "GREEN":
            cvzone.putTextRect(frame, f'Green Time: {green_time:.1f}s', (30, 100),
                               scale=1, thickness=2, offset=5,
                               colorR=signal_color, colorT=(0, 0, 0))

        return frame, len(self.total_count), green_time, signal_state

# Create detector instances for each feed (lazy initialization)
detectors = [None for _ in range(4)]

def get_detector(feed_id):
    """Lazy initialization of detector for specific feed"""
    global detectors, global_optimizer
    if detectors[feed_id] is None:
        print(f"Initializing detector for feed {feed_id}...")
        detector = VehicleDetector()
        detector.lane_id = feed_id  # Set lane ID for phase-based signal management
        detector.optimizer = global_optimizer  # Use shared optimizer
        detectors[feed_id] = detector
    return detectors[feed_id]

def initialize_global_optimizer():
    """Initialize the global optimizer for 4-way junction phase management"""
    global global_optimizer
    if global_optimizer is None:
        print("Initializing global phase-based optimizer...")
        global_optimizer = IntelligentTrafficOptimizer()
    return global_optimizer

# Video source handlers (lazy initialization)
cloud_handler = None
public_handler = None  
live_handler = None

def get_video_handlers():
    """Lazy initialization of video handlers"""
    global cloud_handler, public_handler, live_handler
    if cloud_handler is None:
        print("Initializing video handlers...")
        cloud_handler = CloudVideoHandler()
        public_handler = PublicVideoHandler()
        live_handler = LiveStreamHandler()
    return cloud_handler, public_handler, live_handler

# Global variables for video sources
current_video_sources = [None, None, None, None]
video_source_type = "demo"  # "demo", "cloud", "live", "upload"

def video_processing_thread(feed_id):
    global current_video_sources, public_handler, cloud_handler, live_handler
    
    # Initialize handlers on first run
    try:
        get_video_handlers()
    except Exception as e:
        print(f"Warning: Could not initialize video handlers: {e}")
    
    while True:
        cap = None
        
        # Initialize video source based on type
        if video_source_type == "demo":
            # Use sample/demo videos
            try:
                cap = public_handler.get_sample_video_stream(feed_id % len(public_handler.sample_videos))
            except Exception as e:
                print(f"Error loading demo video for feed {feed_id}: {e}")
                cap = None
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

        detector = get_detector(feed_id)
        processed_frame, count, green_time, signal_state = detector.process_frame(frame)
        
        # Update global optimizer phases to rotate traffic signals properly
        try:
            global_optimizer.update_phase({})
        except Exception as e:
            print(f"Error updating phase: {e}")

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
    """Root endpoint for quick health check"""
    try:
        return jsonify({
            "message": "Traffic Monitor Backend API", 
            "status": "running",
            "port": os.environ.get("PORT", 5000),
            "threads_active": threading.active_count(),
            "environment": "production" if os.environ.get("RENDER") else "development"
        })
    except Exception as e:
        return jsonify({
            "message": "Traffic Monitor Backend API",
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/health')
def health_check():
    """Detailed health check with thread and queue status"""
    queue_sizes = [frame_queues[i].qsize() for i in range(4)]
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "threads_active": threading.active_count(),
        "frame_queue_sizes": queue_sizes,
        "detectors_initialized": [detectors[i] is not None for i in range(4)],
        "video_source_type": video_source_type,
        "handlers_initialized": {
            "cloud": cloud_handler is not None,
            "public": public_handler is not None,
            "live": live_handler is not None
        }
    })

def generate_frames(feed_id):
    while True:
        try:
            # Use timeout to prevent indefinite blocking
            frame_bytes = frame_queues[feed_id].get(timeout=5)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            # Timeout occurred - stream has stalled
            # Send a dummy frame or break to let client reconnect
            print(f"Feed {feed_id} timeout: no frames in queue for 5 seconds")
            break

@app.route('/video_feed/<int:feed_id>')
def video_feed(feed_id):
    if 0 <= feed_id < 4:
        return Response(generate_frames(feed_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Invalid feed ID", 404

@app.route('/get_data/<int:feed_id>')
def get_data(feed_id):
    """Get real-time traffic data for a specific lane"""
    if 0 <= feed_id < 4:
        try:
            # Initialize global optimizer if needed
            initialize_global_optimizer()
            
            data = data_queues[feed_id].get_nowait()
            
            # Add phase information from global optimizer
            lane_phase = "NORTH_SOUTH" if feed_id in [0, 2] else "EAST_WEST"
            
            return jsonify({
                **data,
                "lane_id": feed_id,
                "lane_name": ["North", "East", "South", "West"][feed_id],
                "current_phase": lane_phase,
                "phase_info": "Opposite lanes GREEN" if data["signal_state"] == "RED" else "This lane GREEN"
            })
        except queue.Empty:
            return jsonify({
                "count": 0, 
                "green_time": 30, 
                "signal_state": "GREEN",
                "lane_id": feed_id,
                "lane_name": ["North", "East", "South", "West"][feed_id]
            })
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

@app.route('/get_all_signal_states')
def get_all_signal_states():
    """Get signal state for all 4 lanes (4-way junction phase control)"""
    global global_optimizer
    
    # Initialize optimizer if needed
    initialize_global_optimizer()
    
    # Collect lane metrics from all detectors
    lane_metrics_dict = {}
    for lane_id in range(4):
        detector = get_detector(lane_id)
        try:
            data = data_queues[lane_id].get_nowait()
        except queue.Empty:
            data = {"count": 0, "green_time": 30, "signal_state": "GREEN"}
        
        # Create lane metrics (simplified for phase management)
        lane_metrics_dict[lane_id] = LaneMetrics(
            vehicle_count=data.get("count", 0),
            queue_length=data.get("count", 0) * 5,  # Assume 5m per vehicle
            average_speed=0,
            saturation_level=min(data.get("count", 0) / 30.0, 1.0),
            discharge_rate=2.1,
            arrival_rate=0,
            wait_time_avg=0,
            bottleneck_severity=0
        )
    
    # Get all signal states from global optimizer
    signal_states = global_optimizer.get_all_signal_states(lane_metrics_dict)
    
    # Format response with lane information
    response = {
        "timestamp": time.time(),
        "current_phase": "NORTH_SOUTH" if global_optimizer.current_phase == "NORTH_SOUTH" else "EAST_WEST",
        "lanes": {}
    }
    
    for lane_id in range(4):
        lane_names = ["North", "East", "South", "West"]
        response["lanes"][lane_id] = {
            "lane_name": lane_names[lane_id],
            "signal_state": signal_states[lane_id]["state"],
            "remaining_green_time": signal_states[lane_id]["duration"],
            "phase": signal_states[lane_id]["phase"],
            "vehicle_count": lane_metrics_dict[lane_id].vehicle_count
        }
    
    return jsonify(response)

@app.route('/get_bottleneck_analysis/<int:feed_id>')
def get_bottleneck_analysis(feed_id):
    """Get detailed bottleneck analysis for a specific feed"""
    if 0 <= feed_id < 4:
        detector = detectors[feed_id]
        
        # Get current bottleneck strategies
        strategies = detector.bottleneck_strategies.copy()
        
        # Analyze current traffic conditions
        vehicle_types = [v.vehicle_type for v in detector.vehicles_data[-20:]]  # Last 20 vehicles
        
        analysis = {
            "feed_id": feed_id,
            "current_strategies": strategies,
            "traffic_intensity": len(detector.total_count),
            "signal_optimization": {
                "current_green_time": detector.current_green_time,
                "signal_state": detector.signal_state,
                "vehicles_in_queue": len(detector.vehicles_data),
                "optimization_active": bool(strategies)
            },
            "bottleneck_alerts": []
        }
        
        # Generate bottleneck alerts
        if len(detector.total_count) > 20:
            analysis["bottleneck_alerts"].append({
                "type": "HIGH_CONGESTION",
                "severity": "HIGH",
                "message": f"High vehicle density detected: {len(detector.total_count)} vehicles"
            })
        
        if detector.signal_state == "RED" and (time.time() - detector.signal_start_time) > 60:
            analysis["bottleneck_alerts"].append({
                "type": "LONG_RED_CYCLE", 
                "severity": "MEDIUM",
                "message": "Extended red light may cause spillback"
            })
        
        return jsonify(analysis)
    
    return jsonify({"error": "Invalid feed ID"}), 404

@app.route('/optimize_signal/<int:feed_id>', methods=['POST'])
def optimize_signal(feed_id):
    """Manually trigger signal optimization for a specific feed"""
    if 0 <= feed_id < 4:
        detector = detectors[feed_id]
        
        # Force optimization
        vehicle_types = [v.vehicle_type for v in detector.vehicles_data[-15:]]
        new_green_time = detector.calculate_green_time(len(detector.total_count), vehicle_types)
        
        # Apply optimization
        detector.current_green_time = new_green_time
        detector.signal_start_time = time.time()
        
        return jsonify({
            "message": f"Signal optimized for feed {feed_id}",
            "new_green_time": new_green_time,
            "vehicles_detected": len(detector.total_count),
            "optimization_applied": True
        })
    
    return jsonify({"error": "Invalid feed ID"}), 404

if __name__ == '__main__':
    # Initialize global optimizer for 4-way junction phase management
    initialize_global_optimizer()
    
    # Start video processing threads for each feed
    for i in range(4):
        threading.Thread(target=video_processing_thread, args=(i,), daemon=True).start()

    # Render deployment configuration
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)

# Initialize video threads for production deployment (Render/Heroku)
def initialize_threads():
    """Initialize video processing threads for production"""
    try:
        print("Starting video processing threads...")
        # Initialize global optimizer first
        initialize_global_optimizer()
        
        for i in range(4):
            thread = threading.Thread(target=video_processing_thread, args=(i,), daemon=True)
            thread.start()
            print(f"Started thread for camera {i+1}")
        print("All video threads started successfully")
        print("4-way junction phase-based signal control ACTIVE")
    except Exception as e:
        print(f"Error starting threads: {e}")
        # Continue without threads for basic API functionality

# Start threads when module is imported (for gunicorn)
# Disabled automatic thread startup to allow faster app initialization
# Threads will start on first video request
try:
    print("Flask app module loaded successfully")
    print(f"Environment: {'production' if os.environ.get('RENDER') else 'development'}")
    print(f"Port: {os.environ.get('PORT', 5000)}")
    print("Signal Control: 4-Way Junction Phase-Based Management")
except Exception as e:
    print(f"Error in production initialization: {e}")
