import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import supervision as sv

# --- Vehicle Classifier ---
class VehicleTypeClassifier:
    """
    Lightweight heuristic-based vehicle type classifier.
    Uses YOLO class mappings and bounding box geometry (area, aspect ratio)
    to categorize vehicles into sub-types for real-time edge performance.
    """
    def __init__(self):
        self.classes = ['sedan', 'suv', 'hatchback', 'bus', 'truck', 'others']

    def classify(self, crop, bbox, yolo_class):
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        aspect_ratio = width / height if height > 0 else 1
        area = width * height
        
        # YOLOv8 Class Mappings: 5 = bus, 7 = truck, 2 = car
        if yolo_class == 5: 
            return 'bus', 0.9
        if yolo_class == 7: 
            return 'truck', 0.9
        
        if yolo_class == 2:  # Car class - refine subtype by box dimensions
            if area > 25000: 
                return ('suv', 0.8) if aspect_ratio > 1.7 else ('bus', 0.8)
            if aspect_ratio < 1.4: 
                return 'hatchback', 0.6
            if aspect_ratio > 1.8: 
                return 'suv', 0.6
            return 'sedan', 0.6
            
        return 'others', 0.5

# --- Speed Estimator ---
class EnhancedSpeedEstimator:
    def __init__(self):
        self.classifier = VehicleTypeClassifier()
        self.speed_memory = {}
        self.speed_stable = {}
        self.vehicle_stats = {t: {'count': 0, 'violations': 0} for t in self.classifier.classes}
        self.speed_limits = {
            'sedan': {'warning': 60, 'violation': 100},
            'suv': {'warning': 60, 'violation': 100},
            'hatchback': {'warning': 60, 'violation': 100},
            'bus': {'warning': 60, 'violation': 80},
            'truck': {'warning': 60, 'violation': 90},
            'others': {'warning': 60, 'violation': 100}
        }

    def get_status_color(self, speed, vehicle_type):
        lim = self.speed_limits[vehicle_type]
        if speed > lim['violation']: return (0, 0, 255), "VIOLATION"
        if speed > lim['warning']: return (0, 255, 255), "WARNING"
        return (0, 255, 0), "NORMAL"

    def draw_stats(self, frame):
        # Transparent background box
        overlay = frame.copy()
        log_height = 20 * sum(1 for v in self.vehicle_stats.values() if v['count']) + 10
        cv2.rectangle(overlay, (5, 5), (300, 5 + log_height), (255, 255, 255), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Log text with black color
        y = 25
        for v, s in self.vehicle_stats.items():
            if s['count']:
                text = f"{v.capitalize()}: {s['count']} (Violations: {s['violations']})"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y += 20

    def update_stats(self, vehicle_type, is_violation):
        self.vehicle_stats[vehicle_type]['count'] += 1
        if is_violation:
            self.vehicle_stats[vehicle_type]['violations'] += 1


# --- Input Selection ---
print("Select input:\n1. Webcam\n2. Video File")
choice = input("Choice (1/2): ")

if choice == "1":
    cap = cv2.VideoCapture(0)
    output_path = None
elif choice == "2":
    path = input("Enter video path: ").strip()
    cap = cv2.VideoCapture(path)
    output_path = "enhanced_output_video.mp4"
else:
    print("Invalid choice."); exit()

# --- Calibration Points (Perspective) ---
print("Click 4 points clockwise: TL → TR → BR → BL")
calib_pts = []

def click_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(calib_pts) < 4:
        calib_pts.append([x, y])
        print(f"Point {len(calib_pts)}: ({x}, {y})")

ret, frame = cap.read()
if not ret: print("Failed to grab frame."); exit()

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", click_cb)

while len(calib_pts) < 4:
    img = frame.copy()

    # Draw grid
    h, w = img.shape[:2]
    for i in range(1, 10):
        cv2.line(img, (w * i // 10, 0), (w * i // 10, h), (50, 50, 50), 1)
        cv2.line(img, (0, h * i // 10), (w, h * i // 10), (50, 50, 50), 1)

    # Draw clicked points
    for i, pt in enumerate(calib_pts):
        cv2.circle(img, tuple(pt), 5, (0, 255, 0), -1)
        cv2.putText(img, f"P{i+1}", (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if len(calib_pts) == 4:
        for i in range(4):
            cv2.line(img, tuple(calib_pts[i]), tuple(calib_pts[(i+1)%4]), (255, 255, 0), 2)

    cv2.imshow("Calibration", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
cv2.destroyWindow("Calibration")

SOURCE = np.array(calib_pts, dtype=np.float32)
TARGET = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
M = cv2.getPerspectiveTransform(SOURCE, TARGET)

# --- Scale Points ---
print("Click front and rear of a car (4.5m)")
scale_pts = []

def scale_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(scale_pts) < 2:
        scale_pts.append((x, y))
        print(f"Scale Point {len(scale_pts)}: ({x}, {y})")

cv2.namedWindow("Scale")
cv2.setMouseCallback("Scale", scale_cb)

while len(scale_pts) < 2:
    img = frame.copy()
    if scale_pts:
        cv2.circle(img, scale_pts[0], 5, (0, 255, 0), -1)
        cv2.putText(img, "P1", (scale_pts[0][0] + 10, scale_pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if len(scale_pts) == 2:
        cv2.circle(img, scale_pts[1], 5, (0, 0, 255), -1)
        cv2.putText(img, "P2", (scale_pts[1][0] + 10, scale_pts[1][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.line(img, scale_pts[0], scale_pts[1], (255, 255, 0), 2)

    cv2.imshow("Scale", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
cv2.destroyWindow("Scale")

# --- Calculate Scale ---
p1 = cv2.perspectiveTransform(np.array([[scale_pts[0]]], dtype=np.float32), M)[0][0]
p2 = cv2.perspectiveTransform(np.array([[scale_pts[1]]], dtype=np.float32), M)[0][0]
pixels_per_meter = np.linalg.norm(p1 - p2) / 4.5
print(f"Calibrated: {pixels_per_meter:.2f} px/m")

# --- Setup ---
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(3)), int(cap.get(4))
if output_path:
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

model = YOLO("yolov8s.pt")
tracker = sv.ByteTrack()
estimator = EnhancedSpeedEstimator()
frame_id = 0

# --- Main Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    anchors = detections.get_anchor_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    anchors = cv2.perspectiveTransform(anchors.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)

    labels, colors = [], []
    for i, (tid, p, box, cls, conf) in enumerate(zip(
        detections.tracker_id, anchors, detections.xyxy, detections.class_id, detections.confidence
    )):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        vtype_conf = estimator.classifier.classify(crop, box, cls)
        vtype = vtype_conf[0] if isinstance(vtype_conf, tuple) else vtype_conf


        if tid not in estimator.speed_memory:
            estimator.speed_memory[tid] = deque(maxlen=10)
        estimator.speed_memory[tid].append((p, frame_id / fps))

        if vtype == 'others':
            speed = 0
            estimator.speed_stable[tid] = 0
        else:
            if len(estimator.speed_memory[tid]) >= 2:
                d1, t1 = estimator.speed_memory[tid][0]
                d2, t2 = estimator.speed_memory[tid][-1]
                dist = np.linalg.norm(np.array(d2) - np.array(d1)) / pixels_per_meter
                dt = t2 - t1
                speed = (dist / dt) * 3.6 if dt > 0 else 0
                estimator.speed_stable[tid] = 0.3 * speed + 0.7 * estimator.speed_stable.get(tid, speed)
            speed = estimator.speed_stable.get(tid, 0)

        color, status = estimator.get_status_color(speed, vtype)
        if frame_id % 30 == 0:
            estimator.update_stats(vtype, status == "VIOLATION")

        label = f"{vtype.upper()} {speed:.1f}km/h {status}"
        labels.append(label)
        colors.append(color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Label background rectangle
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)

        # Black label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


    estimator.draw_stats(frame)
    cv2.imshow("Vehicle Speed Estimator", frame)
    if output_path: out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    frame_id += 1

cap.release()
if output_path: out.release()
cv2.destroyAllWindows()
print("Done!")
