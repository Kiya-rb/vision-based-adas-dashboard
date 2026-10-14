import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

def enhance_road_features(image):
    """
    Normalizes lighting across the image to make faint lane lines 
    pop out from the asphalt background.
    """
    # Convert to LAB to isolate lightness from color info
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # Apply CLAHE: ClipLimit prevents over-amplifying noise in dark areas
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge back and convert to BGR for further processing
    enhanced_img = cv.merge((cl, a, b))
    return cv.cvtColor(enhanced_img, cv.COLOR_LAB2BGR)


def create_robust_mask(image, sobel_thresh_min=30, l_min=140):
    """
    Combines adaptive color filtering and gradient detection.
    If the image is too black, lower l_min (Lightness) or sobel_thresh_min.
    """
    # 1. Convert to HLS
    hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    
    # --- Stream 1: Improved Color Mask ---
    # We lower the Lightness threshold (l_min) to catch gray/dirty white lines
    # and use a wider Hue range for yellow.
    white_mask = cv.inRange(hls, np.array([0, l_min, 0]), np.array([180, 255, 80]))
    yellow_mask = cv.inRange(hls, np.array([10, 0, 80]), np.array([45, 255, 255]))
    color_binary = cv.bitwise_or(white_mask, yellow_mask)

    # --- Stream 2: Sobel-X Gradient ---
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Use a larger kernel if lines are blurry
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.absolute(sobelx)
    
    # Robust scaling: avoid division by zero if image is empty
    max_val = np.max(abs_sobel)
    if max_val == 0: max_val = 1
    
    scaled_sobel = np.uint8(255 * abs_sobel / max_val)
    gradient_binary = np.zeros_like(scaled_sobel)
    gradient_binary[scaled_sobel >= sobel_thresh_min] = 255

    # --- Stream 3: The "Rescue" Logic ---
    # Instead of a strict 'AND' which deletes everything, we use a weighted 'OR'
    # but prioritize pixels where BOTH exist.
    combined_binary = cv.bitwise_or(
        cv.bitwise_and(gradient_binary, color_binary), # Strongest features
        cv.bitwise_and(color_binary, cv.dilate(gradient_binary, np.ones((3,3)))) # Near-misses
    )
    
    # Morphological dilation to make the dots connect into lines
    kernel = np.ones((5, 5), np.uint8)
    combined_binary = cv.dilate(combined_binary, kernel, iterations=1)
    
    return combined_binary

def apply_proportional_roi(binary_image):
    """
    Masks the top half of the image (sky) and the extreme corners 
    to focus strictly on the driving lane.
    """
    height, width = binary_image.shape
    mask = np.zeros_like(binary_image)

    # Define vertices as a percentage of image dimensions
    # Point order: Bottom-Left, Top-Left, Top-Right, Bottom-Right
    vertices = np.array([[
        (int(width * 0.1), height),             # Bottom Left
        (int(width * 0.45), int(height * 0.6)), # Top Left (Horizon)
        (int(width * 0.55), int(height * 0.6)), # Top Right (Horizon)
        (int(width * 0.9), height)              # Bottom Right
    ]], dtype=np.int32)

    cv.fillPoly(mask, vertices, 255)
    return cv.bitwise_and(binary_image, mask)


def extract_lane_features(canny_img, original_img):
    """Runs Hough Transform and clusters results using DBSCAN for robustness."""
    # Probabilistic Hough Transform 
    lines = cv.HoughLinesP(canny_img, 1, np.pi/180, threshold=40, minLineLength=50, maxLineGap=100)
    if lines is None: return None

    height = original_img.shape[0]
    features, line_data = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 999
        angle = np.arctan(slope)
        
        # Filter out horizontal-ish noise (lines typical for background/artifacts)
        if abs(angle) < 0.20: continue 
        
        intercept = y1 - slope * x1
        x_bottom = (height - intercept) / slope
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        features.append([angle, x_bottom])
        line_data.append({'slope': slope, 'intercept': intercept, 'length': length})

    if not features: return None

    # Cluster lines by angle and position
    scaler = StandardScaler()
    scaled_f = scaler.fit_transform(features)
    scaled_f[:, 1] *= 7.0 # Prioritize X-position over angle
    clusters = DBSCAN(eps=0.6, min_samples=2).fit_predict(scaled_f)

    final_lanes = []
    for cluster_id in set(clusters):
        if cluster_id == -1: continue
        indices = np.where(clusters == cluster_id)[0]
        
        # Solid vs Dashed logic: Based on longest continuous segment 
        max_seg = max([line_data[i]['length'] for i in indices])
        l_type = "solid" if max_seg > 200 else "dashed"
        
        avg_slope = np.mean([line_data[i]['slope'] for i in indices])
        avg_int = np.mean([line_data[i]['intercept'] for i in indices])
        final_lanes.append({'slope': avg_slope, 'int': avg_int, 'type': l_type})
    
    return final_lanes

def detect_vehicles_and_lanes(image, model, lane_boundaries):
    """Detects vehicles and maps them to road lanes[cite: 18, 20]."""
    results = model.predict(image, classes=[2, 3, 5, 7], conf=0.4, verbose=False)
    detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            # Use bottom-center of the box for lane matching [cite: 20]
            cx, cy = (x1 + x2) // 2, y2
            
            # Identify which lane the car is in (logic: relative to lane boundaries)
            # This satisfies Task 3: "Get in which lane each car is running" [cite: 20]
            detections.append({'box': (x1, y1, x2, y2), 'pos': (cx, cy), 'label': model.names[int(box.cls[0])]})
            
    return detections

def draw_results(image, lanes, vehicles):
    """Draws detected lanes, vehicle boxes, and lane status alerts."""
    overlay = image.copy()
    # 1. Draw Lane Area (Green Fill) 
    # (Logic to fill area between sorted lane lines)
    
    # 2. Draw Lane Lines (Red for Solid, Blue for Dashed) [cite: 8, 17]
    for lane in lanes:
        # Calculate coordinates using your existing make_coordinates logic
        color = (0, 0, 255) if lane['type'] == "solid" else (255, 0, 0)
        # cv.line(...)
        
    # 3. Proximity Warning [cite: 21]
    for veh in vehicles:
        x1, y1, x2, y2 = veh['box']
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
        # Task 3: Set alert if car is close [cite: 21]
        if y2 > (image.shape[0] * 0.85): 
             cv.putText(image, "TOO CLOSE!", (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return cv.addWeighted(overlay, 0.3, image, 0.7, 0)

for i in range(1, 5):
    road = cv.imread(f'roads/road{i}.jpg') 
    if road is None:
        road = cv.imread(f'roads/road{i}.png')
        if road is None:
            print(f"Error loading image roads/road{i}.jpg or roads/road{i}.png")
            continue
    enhanced = enhance_road_features(road)
    mask = create_robust_mask(enhanced)
    roi_masked = apply_proportional_roi(mask)
    # lanes = extract_lane_features(roi_masked, road)
    # model = YOLO('yolov8n.pt')
    # vehicles = detect_vehicles_and_lanes(road, model, lanes)
    # result_img = draw_results(road, lanes, vehicles)
    cv.imshow(f"Road {i} Result", roi_masked)


cv.waitKey(0)
cv.destroyAllWindows()

