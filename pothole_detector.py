import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import time
import os
import cv2
import numpy as np
from collections import deque
from scipy.spatial.distance import cdist
from depth_estimator import DepthEstimator
import json
import requests
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Pothole detection and volume estimation")
    parser.add_argument("--type", type=str, default="video", help="'video' or 'image'")
    parser.add_argument("--path", type=str, required=True, help="Path to the video or image file")
    parser.add_argument("--yolo_model", type=str, default="models/best.pt", help="Path to YOLO model")
    parser.add_argument("--min_confidence", type=float, default=0.5, help="Minimum confidence for YOLO detections")
    parser.add_argument("--tracking_dist_thresh", type=float, default=80, help="Maximum distance (pixels) between centers to associate tracks")
    parser.add_argument("--road_expansion_pixels", type=int, default=10, help="Pixel width of the ring around bbox used for road depth sampling")
    parser.add_argument("--dbscan_eps", type=float, default=75, help="Max distance (pixels) between pothole centers for DBSCAN clustering")
    parser.add_argument("--dbscan_min_samples", type=int, default=2, help="Minimum number of potholes to form a dense cluster core")
    return parser.parse_args()

args = parse_args()
depth_estimator = DepthEstimator(device='cpu')
object_detector = YOLO(args.yolo_model)

# Default intrinsic parameters
fx = 600.0
fy = 600.0
cx = 320.0
cy = 240.0

tracked_potholes = {}
next_pothole_id = 0
TRACKING_BUFFER_LENGTH = 5
next_cluster_id = 0 

def get_current_location():
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        return data.get("lat", 0.0), data.get("lon", 0.0)
    except Exception as e:
        print(f"Error fetching location: {e}")
        return 0.0, 0.0

def assign_track_ids(current_detections, prev_tracked_potholes):
    global next_pothole_id
    current_tracked_potholes_output = {}

    if not prev_tracked_potholes:
        for i, detection in enumerate(current_detections):
            current_tracked_potholes_output[next_pothole_id] = {"bbox": detection, "volumes": deque(maxlen=TRACKING_BUFFER_LENGTH)}
            next_pothole_id += 1
        return current_tracked_potholes_output

    if len(current_detections) == 0:
        return {}

    current_centers = np.array([[(det[0] + det[2]) / 2, (det[1] + det[3]) / 2] for det in current_detections])

    prev_items = list(prev_tracked_potholes.items())
    if not prev_items:
         for i, detection in enumerate(current_detections):
            current_tracked_potholes_output[next_pothole_id] = {"bbox": detection, "volumes": deque(maxlen=TRACKING_BUFFER_LENGTH)}
            next_pothole_id += 1
         return current_tracked_potholes_output

    prev_track_ids = [item[0] for item in prev_items]
    prev_centers = np.array([[(data["bbox"][0] + data["bbox"][2]) / 2, (data["bbox"][1] + data["bbox"][3]) / 2] for _, data in prev_items])

    distances = cdist(current_centers, prev_centers)

    used_prev_indices = set()
    used_current_indices = set()
    potential_matches = []
    for i in range(len(current_detections)):
        for j in range(len(prev_track_ids)):
            potential_matches.append((i, j, distances[i, j]))

    potential_matches.sort(key=lambda x: x[2])

    for current_idx, prev_idx, dist in potential_matches:
        if dist < args.tracking_dist_thresh:
            if current_idx not in used_current_indices and prev_idx not in used_prev_indices:
                track_id = prev_track_ids[prev_idx]
                current_tracked_potholes_output[track_id] = {
                    "bbox": current_detections[current_idx],
                    "volumes": prev_tracked_potholes[track_id]["volumes"]
                }
                used_current_indices.add(current_idx)
                used_prev_indices.add(prev_idx)
        else:
            break

    for i, detection in enumerate(current_detections):
        if i not in used_current_indices:
            current_tracked_potholes_output[next_pothole_id] = {"bbox": detection, "volumes": deque(maxlen=TRACKING_BUFFER_LENGTH)}
            next_pothole_id += 1

    return current_tracked_potholes_output

def estimate_real_world_volume_v2(bbox, full_depth_map):
    MIDAS_INVERSE_DEPTH_SCALE = 500.0
    RELATIVE_DEPTH_DIFF_TO_METERS = 0.05

    x1, y1, x2, y2 = map(int, bbox)
    pothole_width_pixels = x2 - x1
    pothole_height_pixels = y2 - y1

    if pothole_width_pixels <= 0 or pothole_height_pixels <= 0:
        print("Warning: Invalid bounding box dimensions.")
        return 0

    # Use only the pothole depth values
    h, w = full_depth_map.shape
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(w, x2), min(h, y2)

    if ix1 >= ix2 or iy1 >= iy2:
        print("Warning: Invalid pothole bounding box.")
        return 0

    pothole_mask = np.zeros_like(full_depth_map, dtype=bool)
    pothole_mask[iy1:iy2, ix1:ix2] = True

    pothole_depth_values = full_depth_map[pothole_mask]

    if pothole_depth_values.size == 0:
        print("Warning: No valid depth values for pothole area.")
        return 0

    avg_pothole_depth_relative = np.mean(pothole_depth_values)

    if not np.isfinite(avg_pothole_depth_relative):
        print("Warning: Non-finite average pothole depth.")
        return 0

    # Calculate the real-world dimensions and volume
    distance_z = MIDAS_INVERSE_DEPTH_SCALE / avg_pothole_depth_relative

    real_width_meters = (pothole_width_pixels * distance_z) / fx
    real_height_meters = (pothole_height_pixels * distance_z) / fy

    real_pothole_depth_meters = avg_pothole_depth_relative * RELATIVE_DEPTH_DIFF_TO_METERS
    real_pothole_depth_meters = max(0, real_pothole_depth_meters)

    volume_m3 = real_width_meters * real_height_meters * real_pothole_depth_meters
    volume_liters = volume_m3 * 10

    return max(0, volume_liters)

def save_detection_data(detection_data, output_file="data/detections.json"):
    try:
        # Load existing data if the file exists and is valid
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {output_file} is empty or corrupted. Reinitializing it.")
                    existing_data = []
        else:
            existing_data = []

        # Append new detection data to the existing data
        existing_data.extend(detection_data)

        # Convert all NumPy types to Python native types
        for entry in existing_data:
            for key, value in entry.items():
                if isinstance(value, (np.float32, np.float64)):
                    entry[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    entry[key] = int(value)
                elif isinstance(value, list):
                    entry[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, (np.int32, np.int64)) else v for v in value]

        # Save the updated data back to the file
        with open(output_file, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(f"Detection data saved to {output_file}")
    except Exception as e:
        print(f"Error saving detection data: {e}")

def detect_and_display(detection_type, path=None):
    global tracked_potholes, next_pothole_id, fx, fy, cx, cy

    if detection_type == 'image':
        frame = cv2.imread(path)
        if frame is None:
            print(f"Error: Could not read image at {path}")
            return

        tracked_potholes = {}
        next_pothole_id = 0

        full_depth_map = depth_estimator.estimate_depth(frame)
        if full_depth_map is None or full_depth_map.size == 0:
            print("Error: Failed to estimate depth for the image.")
            cv2.imshow("Pothole Detection - No Depth", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        results = object_detector(frame, conf=args.min_confidence, verbose=False)

        frame_with_detections = draw_detections_with_tracking(frame.copy(), results, full_depth_map, path)

        cv2.imshow("Pothole Detection", frame_with_detections)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif detection_type == 'video':
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {path}")
            return

        tracked_potholes = {}
        next_pothole_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            start_time = time.time()

            full_depth_map = depth_estimator.estimate_depth(frame)
            if full_depth_map is None or full_depth_map.size == 0:
                print("Warning: Failed to estimate depth for current frame. Skipping volume calculation.")
                full_depth_map = np.zeros(frame.shape[:2], dtype=np.float32)

            results = object_detector(frame, conf=args.min_confidence, verbose=False)

            frame_with_detections = draw_detections_with_tracking(frame.copy(), results, full_depth_map, path)

            end_time = time.time()
            fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            cv2.putText(frame_with_detections, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Pothole Detection", frame_with_detections)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif detection_type == 'realtime':
        cap = cv2.VideoCapture(0)  # Open webcam
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        tracked_potholes = {}
        next_pothole_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            start_time = time.time()

            full_depth_map = depth_estimator.estimate_depth(frame)
            if full_depth_map is None or full_depth_map.size == 0:
                print("Warning: Failed to estimate depth for current frame. Skipping volume calculation.")
                full_depth_map = np.zeros(frame.shape[:2], dtype=np.float32)

            results = object_detector(frame, conf=args.min_confidence, verbose=False)

            frame_with_detections = draw_detections_with_tracking(frame.copy(), results, full_depth_map, "realtime")

            end_time = time.time()
            fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            cv2.putText(frame_with_detections, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Pothole Detection - Realtime", frame_with_detections)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print(f"Error: Unknown detection type '{detection_type}'. Use 'image', 'video', or 'realtime'.")

def draw_detections_with_tracking(frame, results, full_depth_map, image_path):
    global tracked_potholes

    detections = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    pothole_indices = np.where((class_ids == 0) & (confidences >= args.min_confidence))[0]
    current_pothole_detections = detections[pothole_indices]
    current_confidences = confidences[pothole_indices]

    tracked_potholes = assign_track_ids(current_pothole_detections, tracked_potholes)

    cluster_map = {}
    cluster_bboxes = {}
    detection_data = []

    # Create a folder to save pothole images
    os.makedirs("pothole_images", exist_ok=True)

    active_track_items = list(tracked_potholes.items())

    if active_track_items:
        track_ids = [item[0] for item in active_track_items]
        centers = np.array([[(data["bbox"][0] + data["bbox"][2]) / 2, (data["bbox"][1] + data["bbox"][3]) / 2] for _, data in active_track_items])

        if len(centers) > 0:
            db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples).fit(centers)
            labels = db.labels_

            for i, track_id in enumerate(track_ids):
                cluster_id = labels[i]
                cluster_map[track_id] = cluster_id
        else:
            for track_id in track_ids:
                cluster_map[track_id] = -1

    frame_height, frame_width = frame.shape[:2]
    latitude, longitude = get_current_location()

    # Draw individual pothole bounding boxes
    for i, (track_id, track_data) in enumerate(active_track_items):
        bbox = track_data["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(frame_width, x2), min(frame_height, y2)

        if x1c >= x2c or y1c >= y2c:
            continue

        volume = estimate_real_world_volume_v2(bbox, full_depth_map)
        track_data["volumes"].append(volume)
        avg_volume = np.mean(track_data["volumes"]) if track_data["volumes"] else 0

        cluster_id = cluster_map.get(track_id, -1)

        # Draw individual pothole bounding box
        color = (0, 255, 0) if cluster_id == -1 else (255, 255, 0)  # Green for unclustered, Yellow for clustered
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)

        # Add label with confidence and volume
        confidence = float(current_confidences[i]) if i < len(current_confidences) else 0
        label = f"ID:{track_id} Conf:{confidence:.2f} Vol:{avg_volume:.2f}"
        label_pos = (x1c, y1c - 10 if y1c > 20 else y1c + 15)
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save the cropped pothole image
        pothole_image = frame[y1c:y2c, x1c:x2c]
        pothole_image_path = os.path.join("pothole_images", f"pothole_{track_id}.jpg")
        cv2.imwrite(pothole_image_path, pothole_image)

        # Add to cluster bounding boxes
        if cluster_id != -1:
            if cluster_id not in cluster_bboxes:
                cluster_bboxes[cluster_id] = []
            cluster_bboxes[cluster_id].append(bbox)

        # Calculate additional details for JSON entry
        height = y2c - y1c
        width = x2c - x1c
        area = height * width
        depth = avg_volume / area if area > 0 else 0

        # Create detection entry
        detection_entry = {
            "id": track_id,  # Add unique ID for the pothole
            "timestamp": datetime.now().isoformat(),
            "latitude": latitude,
            "longitude": longitude,
            "bbox": [int(x1c), int(y1c), int(x2c), int(y2c)],
            "height": int(height),
            "width": int(width),
            "depth": depth,
            "area": int(area),
            "volume": avg_volume,
            "image_path": pothole_image_path,
            "confidence": confidence,
            "cluster_id": cluster_id if cluster_id != -1 else None
        }
        detection_data.append(detection_entry)

    # Draw cluster bounding boxes
    for cluster_id, bboxes in cluster_bboxes.items():
        if cluster_id == -1:
            continue

        # Calculate the bounding box that encompasses all potholes in the cluster
        all_coords = np.array([[b[0], b[1], b[2], b[3]] for b in bboxes])
        min_x = int(np.min(all_coords[:, 0]))
        min_y = int(np.min(all_coords[:, 1]))
        max_x = int(np.max(all_coords[:, 2]))
        max_y = int(np.max(all_coords[:, 3]))

        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(frame_width, max_x), min(frame_height, max_y)

        # Draw the cluster bounding box
        cluster_color = (255, 0, 255)  # Magenta color for cluster bounding box
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), cluster_color, 3)

        # Label the cluster
        cluster_label = f"Cluster {cluster_id}"
        label_pos = (min_x, min_y - 10 if min_y > 20 else min_y + 15)
        cv2.putText(frame, cluster_label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, cluster_color, 2)

    # Save detection data to JSON
    save_detection_data(detection_data)
    return frame

if __name__ == "__main__":
    if depth_estimator is None:
        print("Error: Depth Estimator failed to initialize.")
    else:
        detect_and_display(args.type, args.path)