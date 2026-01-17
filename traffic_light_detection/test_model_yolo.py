import os
# Force CPU mode (Good for RPi)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

from ultralytics import YOLO
import cv2
import numpy as np
import time
from typing import List, Tuple
import argparse

class TrafficLightDetectorYOLO:
    def __init__(self, model_path: str):
        """
        Initialize YOLO traffic light detector
        """
        self.model_path = model_path
        print(f"Loading model from {self.model_path}...")
        
        # Load the model
        self.model = YOLO(self.model_path)
        
        self.names = self.model.names 
        print(f"Model loaded. Classes detected: {self.names}")
        
        # Define colors dynamically based on class names
        # Default to white, but try to match common names
        self.colors = {}
        for cls_id, name in self.names.items():
            name = name.lower()
            if 'green' in name:
                self.colors[cls_id] = (0, 255, 0)   # Green
            elif 'red' in name:
                self.colors[cls_id] = (0, 0, 255)   # Red (BGR format in OpenCV)
            elif 'yellow' in name:
                self.colors[cls_id] = (0, 255, 255) # Yellow
            else:
                self.colors[cls_id] = (255, 255, 255) # White for others

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5):
        """
        Detect traffic lights in a frame
        """
        
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        
        if results[0].boxes:
            for box in results[0].boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                detections.append((x1, y1, x2, y2, conf, cls_id))
        
        return detections

    def draw_detections(self, image: np.ndarray, detections: List):
        img_draw = image.copy()
        
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get color and label
            color = self.colors.get(cls_id, (255, 255, 255))
            label_name = self.names[cls_id]
            
            # Draw box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{label_name}: {conf:.2f}"
            cv2.putText(img_draw, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_draw

# ==========================================
# NEW FUNCTION: Real-Time Camera Loop (For RPi)
# ==========================================
def run_live_camera(model_path, source=0):
    detector = TrafficLightDetectorYOLO(model_path)
    
    # Open Camera (0 for webcam, or video path)
    cap = cv2.VideoCapture(source)
    
    # Set Camera Resolution (Lower = Faster on RPi)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting video stream. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        
        # 1. Detect
        detections = detector.detect(frame, conf_threshold=0.4)
        
        # 2. Robot Logic (The "University Project" part)
        action = "GO" # Default
        for det in detections:
            cls_id = det[5]
            class_name = detector.names[cls_id].lower()
            
            # Simple logic: If ANY red light is found, STOP.
            if 'red' in class_name:
                action = "STOP"
                break # Priority stop
        
        # 3. Draw
        result_frame = detector.draw_detections(frame, detections)
        
        # Calculate FPS
        fps = 1 / (time.time() - start)
        
        # Display Status
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_frame, f"ROBOT: {action}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if action == "STOP" else (0, 255, 0), 2)
        
        cv2.imshow('Robot View', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
def test_image_file(model_path: str, image_path: str):
    """
    Test the model on a single image file and display results.
    """
    # 1. Initialize Detector
    detector = TrafficLightDetectorYOLO(model_path)
    
    # 2. Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'. Check the path.")
        return

    print(f"Testing on image: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # 3. Run Detection & Measure Speed
    start_time = time.time()
    
    # Run detection
    detections = detector.detect(image, conf_threshold=0.3)
    
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000 # Convert to milliseconds

    # 4. Print detailed results to console (Good for reports)
    print(f"\n--- Performance ---")
    print(f"Inference Time: {inference_time:.1f} ms")
    print(f"Detections Found: {len(detections)}")
    
    for i, (x1, y1, x2, y2, conf, cls_id) in enumerate(detections):
        class_name = detector.names[cls_id]
        print(f"   {i+1}. {class_name} | Confidence: {conf:.1%} | Loc: [{int(x1)}, {int(y1)}]")

    # 5. Draw and Save
    result_image = detector.draw_detections(image, detections)
    
    # Construct output filename (e.g., test.jpg -> test_result.jpg)
    filename, ext = os.path.splitext(image_path)
    output_path = f"{filename}_result{ext}"
    
    cv2.imwrite(output_path, result_image)
    print(f"Result image saved to: {output_path}")

    # 6. Display (Press any key to close)
    cv2.imshow("Traffic Light Detection", result_image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traffic Light Detection with YOLOv8')
    
   
    parser.add_argument('--source', type=str, default='0',
                       help='Image/Video path or webcam index (default: 0)')

    args = parser.parse_args()
    
    # To test an image:
    
    # test_image_file("best.pt", args.source)
    
    # To run live (Laptop Webcam):
    run_live_camera("best.pt", source=0)