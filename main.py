import numpy as np
import cv2
import mediapipe as mp
import time
import json
from datetime import datetime
from collections import deque

# ============== METRICS TRACKER ==============
class MetricsTracker:
    def __init__(self, window_size=30):
        self.session_start = time.time()
        self.frame_times = deque(maxlen=window_size)  # Rolling window for FPS
        self.inference_times = deque(maxlen=window_size)
        
        # Counters
        self.total_frames = 0
        self.frames_with_face = 0
        
        # Distraction tracking
        self.distraction_events = []  # List of {start, end, duration}
        self.current_distraction_start = None
        self.is_distracted = False
        self.total_distracted_frames = 0
        self.total_focused_frames = 0
        
        # For logging
        self.log_data = {
            "session_start": datetime.now().isoformat(),
            "frames": [],
            "distraction_events": [],
            "summary": {}
        }
    
    def start_frame(self):
        """Call at start of frame processing"""
        self.frame_start = time.time()
    
    def record_inference(self, inference_time):
        """Record inference time in ms"""
        self.inference_times.append(inference_time * 1000)  # Convert to ms
    
    def end_frame(self, face_detected, is_looking_away):
        """Call at end of frame processing"""
        frame_time = time.time() - self.frame_start
        self.frame_times.append(frame_time)
        self.total_frames += 1
        
        if face_detected:
            self.frames_with_face += 1
            
            # Track distraction state transitions
            if is_looking_away:
                self.total_distracted_frames += 1
                if not self.is_distracted:
                    # Started being distracted
                    self.is_distracted = True
                    self.current_distraction_start = time.time()
            else:
                self.total_focused_frames += 1
                if self.is_distracted:
                    # Stopped being distracted
                    self.is_distracted = False
                    if self.current_distraction_start:
                        duration = time.time() - self.current_distraction_start
                        self.distraction_events.append({
                            "start": self.current_distraction_start - self.session_start,
                            "end": time.time() - self.session_start,
                            "duration": duration
                        })
                        self.current_distraction_start = None
    
    def get_fps(self):
        """Get current FPS (rolling average)"""
        if len(self.frame_times) == 0:
            return 0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))
    
    def get_avg_inference_ms(self):
        """Get average inference time in ms"""
        if len(self.inference_times) == 0:
            return 0
        return sum(self.inference_times) / len(self.inference_times)
    
    def get_face_detection_rate(self):
        """Get percentage of frames with face detected"""
        if self.total_frames == 0:
            return 0
        return (self.frames_with_face / self.total_frames) * 100
    
    def get_focus_percentage(self):
        """Get percentage of time focused (when face was detected)"""
        total = self.total_focused_frames + self.total_distracted_frames
        if total == 0:
            return 100
        return (self.total_focused_frames / total) * 100
    
    def get_session_summary(self):
        """Get complete session summary"""
        session_duration = time.time() - self.session_start
        
        # Close any ongoing distraction event
        if self.is_distracted and self.current_distraction_start:
            self.distraction_events.append({
                "start": self.current_distraction_start - self.session_start,
                "end": session_duration,
                "duration": time.time() - self.current_distraction_start
            })
        
        avg_distraction_duration = 0
        if self.distraction_events:
            avg_distraction_duration = sum(e["duration"] for e in self.distraction_events) / len(self.distraction_events)
        
        return {
            "session_duration_sec": round(session_duration, 2),
            "total_frames": self.total_frames,
            "avg_fps": round(self.get_fps(), 1),
            "avg_inference_ms": round(self.get_avg_inference_ms(), 2),
            "face_detection_rate": round(self.get_face_detection_rate(), 1),
            "focus_percentage": round(self.get_focus_percentage(), 1),
            "distraction_events_count": len(self.distraction_events),
            "avg_distraction_duration_sec": round(avg_distraction_duration, 2),
            "distraction_events": self.distraction_events
        }
    
    def save_log(self, filename=None):
        """Save session log to JSON file"""
        if filename is None:
            filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.log_data["session_end"] = datetime.now().isoformat()
        self.log_data["summary"] = self.get_session_summary()
        self.log_data["distraction_events"] = self.distraction_events
        
        with open(filename, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        
        return filename


# ============== MAIN APPLICATION ==============

# Initialize MediaPipe Face Mesh
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(max_num_faces=1)

mpDrawing = mp.solutions.drawing_utils
drawingSpec = mpDrawing.DrawingSpec(thickness=1, circle_radius=1)

# Start webcam
cap = cv2.VideoCapture(0)

# Eye landmark indices
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398]

# Distraction detection params
DISTRACTION_THRESHOLD_PX = 50  # Pixels from center
DISTRACTION_FRAME_THRESHOLD = 60  # Frames before triggering alert
distraction_counter = 0

# Initialize metrics tracker
metrics = MetricsTracker()

def get_eye_bbox(eye_landmarks):
    x_coords = [point[0] for point in eye_landmarks]
    y_coords = [point[1] for point in eye_landmarks]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def is_circle(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return 0.5 < circularity < 1.2

def draw_metrics_overlay(frame, metrics_tracker, is_distracted):
    """Draw real-time metrics on frame"""
    fps = metrics_tracker.get_fps()
    inference_ms = metrics_tracker.get_avg_inference_ms()
    focus_pct = metrics_tracker.get_focus_percentage()
    detection_rate = metrics_tracker.get_face_detection_rate()
    
    # Background for metrics
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Text styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), font, 0.5, color, 1)
    cv2.putText(frame, f"Inference: {inference_ms:.1f}ms", (20, 55), font, 0.5, color, 1)
    cv2.putText(frame, f"Face Detect: {detection_rate:.0f}%", (20, 75), font, 0.5, color, 1)
    cv2.putText(frame, f"Focus: {focus_pct:.0f}%", (20, 95), font, 0.5, color, 1)
    
    # Distraction indicator
    if is_distracted:
        cv2.rectangle(frame, (frame.shape[1]-150, 10), (frame.shape[1]-10, 50), (0, 0, 200), -1)
        cv2.putText(frame, "DISTRACTED", (frame.shape[1]-140, 37), font, 0.6, (255, 255, 255), 2)

print("\n" + "="*50)
print("FOCALLUS - Distraction Monitoring")
print("="*50)
print("Press 'Q' to quit and see session summary")
print("="*50 + "\n")

while cap.isOpened():
    metrics.start_frame()
    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Time the inference
    inference_start = time.time()
    results = face_mesh.process(rgb_frame)
    inference_time = time.time() - inference_start
    metrics.record_inference(inference_time)
    
    face_detected = False
    is_looking_away = False

    if results.multi_face_landmarks:
        face_detected = True
        for face_landmarks in results.multi_face_landmarks:
            mpDrawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawingSpec,
                connection_drawing_spec=drawingSpec)

            h, w, _ = frame.shape
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayFrame = cv2.equalizeHist(grayFrame)
            grayFrame = cv2.GaussianBlur(grayFrame, (7, 7), 0)

            for eye in [left_eye, right_eye]:
                x1, y1, x2, y2 = get_eye_bbox(eye)
                eye_region = grayFrame[y1:y2, x1:x2]

                threshold = cv2.adaptiveThreshold(
                    eye_region, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11, 2
                )

                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 30 < area < 300 and is_circle(contour):
                        (px, py), radius = cv2.minEnclosingCircle(contour)
                        center = (int(px) + x1, int(py) + y1)
                        cv2.circle(frame, center, int(radius), (255, 0, 0), 2)

            center_x = (left_eye[0][0] + right_eye[0][0]) // 2  
            frame_center = frame.shape[1] // 2                 

            if abs(center_x - frame_center) > DISTRACTION_THRESHOLD_PX:  
                distraction_counter += 1
                is_looking_away = True
            else:
                distraction_counter = 0
                is_looking_away = False

            if distraction_counter > DISTRACTION_FRAME_THRESHOLD:  
                pass  # Alert handled by metrics overlay now

    # Record metrics for this frame
    metrics.end_frame(face_detected, is_looking_away and distraction_counter > DISTRACTION_FRAME_THRESHOLD)
    
    # Draw overlay
    frame = cv2.resize(frame, (640, 360))
    draw_metrics_overlay(frame, metrics, distraction_counter > DISTRACTION_FRAME_THRESHOLD)
    
    cv2.imshow("Focallus - Distraction Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ============== SESSION SUMMARY ==============
summary = metrics.get_session_summary()

print("\n" + "="*50)
print("SESSION SUMMARY")
print("="*50)
print(f"Duration:              {summary['session_duration_sec']:.1f} seconds")
print(f"Total Frames:          {summary['total_frames']}")
print(f"Average FPS:           {summary['avg_fps']:.1f}")
print(f"Avg Inference Latency: {summary['avg_inference_ms']:.2f} ms")
print(f"Face Detection Rate:   {summary['face_detection_rate']:.1f}%")
print("-"*50)
print(f"Focus Percentage:      {summary['focus_percentage']:.1f}%")
print(f"Distraction Events:    {summary['distraction_events_count']}")
if summary['distraction_events_count'] > 0:
    print(f"Avg Distraction Time:  {summary['avg_distraction_duration_sec']:.2f} seconds")
print("="*50)

# Save log file
log_file = metrics.save_log()
print(f"\nSession log saved to: {log_file}")
