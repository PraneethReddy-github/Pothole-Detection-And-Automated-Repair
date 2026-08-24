import cv2
import numpy as np
import math

def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=50)
    return edges, lines

def average_lane_angle(lines):
    angles = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    angles = [a for a in angles if -80 < a < 80]  # Only consider reasonable angles
    if len(angles) == 0:
        return None
    avg_angle = np.mean(angles)
    return avg_angle

def calculate_focal_length(camera_height_m, camera_angle_deg, pixel_to_meter_ratio):
    angle_rad = math.radians(camera_angle_deg)
    fy = (camera_height_m / pixel_to_meter_ratio) / math.tan(angle_rad)
    fx = fy  # Assuming square pixels (same focal length for both x and y)
    return fx, fy

def draw_lanes(image, lines):
    lane_image = np.copy(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return lane_image

def main(image_path, camera_height_m, camera_angle_deg, real_world_patch_width_m):
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    edges, lines = detect_lanes(image.copy())
    if lines is None:
        print("No lanes detected!")
        return

    avg_angle = average_lane_angle(lines)
    if avg_angle is not None:
        print(f"Estimated Road Angle: {avg_angle:.2f} degrees")
    else:
        print("Could not estimate road angle.")
        avg_angle = 0  # Default angle if not detectable

    x1, y1, x2, y2 = lines[0][0]  # Use the first detected line for pixel distance
    pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    pixel_to_meter_ratio = real_world_patch_width_m / pixel_distance
    fx, fy = calculate_focal_length(camera_height_m, camera_angle_deg, pixel_to_meter_ratio)
    cx = image_width / 2
    cy = image_height / 2

    print(f"fx = {fx:.2f} px")
    print(f"fy = {fy:.2f} px")
    print(f"cx = {cx:.2f} px")
    print(f"cy = {cy:.2f} px")
    print(f"Pixel to Meter Ratio = {pixel_to_meter_ratio:.6f} m/px")

    # Draw the lanes and display the result
    lane_image = draw_lanes(image.copy(), lines)
    cv2.imshow("Detected Lanes", lane_image)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "sample_data/images/pothole_0.jpg"  # Replace with your image path
    camera_height_m = 1.5
    camera_angle_deg = 20
    real_world_patch_width_m = 3.5  # The real-world width of the patch used for calculation
    main(image_path, camera_height_m, camera_angle_deg, real_world_patch_width_m)
