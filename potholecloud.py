import os
import cv2
import time
import threading
from io import BytesIO
from PIL import Image
from google.cloud import storage
from ultralytics import YOLO
import geocoder

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'pothole.json'
BUCKET_NAME = 'pothole_img_storage'

# Initialize Google Cloud Storage
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# Load YOLO model
model = YOLO('models/best.pt')

# Debounce tracking
last_detection_time = 0
last_location = (None, None)

# Function: Get current location
def get_location():
    g = geocoder.ip('me')
    if g.ok:
        return g.latlng[0], g.latlng[1], g.address if g.address else "Unknown Location"
    return None, None, "Location not available"

# Function: Check if detection is duplicate
def is_duplicate_detection(current_time, lat, lon, threshold_seconds=10, threshold_distance=0.0005):
    global last_detection_time, last_location
    if abs(current_time - last_detection_time) < threshold_seconds:
        if last_location != (None, None):
            if abs(lat - last_location[0]) < threshold_distance and abs(lon - last_location[1]) < threshold_distance:
                return True
    last_detection_time = current_time
    last_location = (lat, lon)
    return False

# Function: Upload image asynchronously
def upload_async(image_array, filename, metadata):
    success, buffer = cv2.imencode('.jpg', image_array)
    if success:
        blob = bucket.blob(filename)
        if metadata:
            blob.metadata = metadata
        blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
        print(f"[UPLOAD] Uploaded {filename} to GCS")

# Function: Process frame and handle detection
def process_frame(frame):
    results = model(frame)
    potholes_detected = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0]) * 100
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{confidence:.1f}%'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            potholes_detected = True

    if potholes_detected:
        current_time = time.time()
        latitude, longitude, location_address = get_location()

        if is_duplicate_detection(current_time, latitude, longitude):
            print("[SKIP] Duplicate detection.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_filename = f'pothole_{timestamp}.jpg'

        metadata = {
            'timestamp': timestamp,
            'latitude': str(latitude),
            'longitude': str(longitude),
            'location': location_address
        }

        threading.Thread(target=upload_async, args=(frame.copy(), image_filename, metadata)).start()

# Function: Process image
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is not None:
        process_frame(frame)
    else:
        print("[ERROR] Could not read image.")

# Function: Process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        cv2.imshow('Video Pothole Detection - Press "q" to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function: Process webcam
def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not found.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break
        process_frame(frame)
        cv2.imshow('Real-time Pothole Detection - Press "q" to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Input selection
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
        process_webcam()
    else:
        print("[ERROR] Invalid choice.")

if __name__ == '__main__':
    main()
