# 🚧 Pothole Detection Project: Part 1 & 2 - Detection, Clustering & Volume Estimation

import os
import cv2
import time
import json
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from sklearn.cluster import DBSCAN

# === Setup ===
os.makedirs('temp', exist_ok=True)
os.makedirs('data', exist_ok=True)

model = YOLO('models/best.pt')  # Load YOLO model
detections_file = 'data/detections.json'
detections = []  # This will hold our data

# === Globals for simulation and tracking ===
dummy_distance = 0  # Simulated distance in meters
frame_counter = 0   # Frame counter for simulating movement
pothole_count = 0   # Count of unique potholes
previous_detections = []  # Memory for duplicate detection

# === Dummy location provider with distance simulation ===
def get_dummy_location():
    global dummy_distance, frame_counter
    frame_counter += 1
    if frame_counter >= 10:
        dummy_distance += 1
        frame_counter = 0
    return dummy_distance, dummy_distance  # Simulated (lat, lon)

# === Clear detection memory every 10 meters to avoid stale duplicates ===
def maybe_reset_detection_memory():
    if dummy_distance % 10 == 0 and frame_counter == 0:
        previous_detections.clear()

# === Save detections to JSON ===
def save_detections():
    with open(detections_file, 'w') as f:
        json.dump(detections, f, indent=4)

# === Calculate area of a bounding box ===
def calculate_area(bbox):
    width = bbox["x2"] - bbox["x1"]
    height = bbox["y2"] - bbox["y1"]
    return width * height

# === Check if detection is duplicate based on spatial distance ===
def is_duplicate(new_box, threshold=40):
    nx, ny = (new_box['x1'] + new_box['x2']) / 2, (new_box['y1'] + new_box['y2']) / 2
    for prev in previous_detections:
        px, py = (prev['x1'] + prev['x2']) / 2, (prev['y1'] + prev['y2']) / 2
        distance = np.sqrt((px - nx)**2 + (py - ny)**2)
        if distance < threshold:
            return True
    return False

# === Process a single frame for pothole detection ===
def process_frame(frame):
    global previous_detections, pothole_count
    maybe_reset_detection_memory()

    results = model(frame, verbose=False)
    frame_detections = []
    current_frame_boxes = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            new_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            if is_duplicate(new_box):
                continue

            pothole_count += 1  # Unique pothole count
            label = f"{confidence*100:.1f}% #{pothole_count}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            timestamp = datetime.now().isoformat()
            lat, lon = get_dummy_location()

            detection = {
                "timestamp": timestamp,
                "latitude": lat,
                "longitude": lon,
                "bbox": new_box,
                "confidence": confidence,
                "area": calculate_area(new_box)
            }
            frame_detections.append(detection)
            current_frame_boxes.append(new_box)

    previous_detections.extend(current_frame_boxes)

    if frame_detections:
        cluster_and_group(frame_detections, frame)

# === Cluster detections in a frame using DBSCAN ===
def cluster_and_group(detections_in_frame, frame):
    coords = np.array([
        [(d['bbox']['x1'] + d['bbox']['x2']) / 2, (d['bbox']['y1'] + d['bbox']['y2']) / 2]
        for d in detections_in_frame
    ])
    clustering = DBSCAN(eps=50, min_samples=1).fit(coords)

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        clusters.setdefault(label, []).append(detections_in_frame[idx])

    for cluster_id, items in clusters.items():
        total_area = sum(item['area'] for item in items)
        estimated_volume = total_area * 0.05
        print(f"[CLUSTER {cluster_id}] Total Area: {total_area:.2f} px² - Count: {len(items)}")
        print(f"→ Estimated Volume of Tar Needed: {estimated_volume:.2f} liters")

        x_min = min(item['bbox']['x1'] for item in items)
        y_min = min(item['bbox']['y1'] for item in items)
        x_max = max(item['bbox']['x2'] for item in items)
        y_max = max(item['bbox']['y2'] for item in items)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(frame, f"Cluster {cluster_id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for item in items:
            item['image_path'] = None
            detections.append(item)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cluster_img_path = f"temp/clusters_combined_{timestamp}.jpg"
    cv2.imwrite(cluster_img_path, frame)
    print(f"[SAVED] Combined cluster image saved at {cluster_img_path}")

    save_detections()

# === Webcam input ===
def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame)
        cv2.imshow("Pothole Detection - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Image input ===
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is not None:
        process_frame(frame)
        cv2.imshow("Image Pothole Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[ERROR] Could not read image.")

# === Video input ===
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        cv2.imshow("Video Pothole Detection - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# === User input selection ===
def main():
    print("Choose input type:")
    print("1. Image")
    print("2. Video")
    print("3. Real-time webcam")

    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        image_path = input("Enter the path to the image file: ")
        if os.path.exists(image_path):
            process_image(image_path)
        else:
            print("[ERROR] File not found.")
    elif choice == '2':
        video_path = input("Enter the path to the video file: ")
        if os.path.exists(video_path):
            process_video(video_path)
        else:
            print("[ERROR] File not found.")
    elif choice == '3':
        run_camera()
    else:
        print("[ERROR] Invalid choice.")

# === Run ===
if __name__ == '__main__':
    main()
