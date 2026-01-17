import os
import cv2
import numpy as np
import time
from typing import List, Tuple
import tflite_runtime.interpreter as tflite


class TFLiteTrafficLightDetector:
    def __init__(self, tflite_model_path: str):
        print(f"Loading TFLite model: {tflite_model_path}")
        
        self.interpreter = tflite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Model input size (should be 640 from your script)
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        
     
        self.class_names = {0: 'Green', 1: 'Red', 2: 'Yellow'} 
        self.colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
        
        

    def postprocess_output(self, output_data: np.ndarray, 
                          img_h: int, img_w: int,
                          conf_threshold: float = 0.5) -> List[Tuple]:
        detections = []
        
        # YOLOv8 shape from your converter: [1, 7, 8400] 
        # (7 columns = 4 box coords + 3 class scores)
        output = np.transpose(np.squeeze(output_data))
        
        # Scaling factors: How to get from 640x640 space to your Camera resolution
        x_scale = img_w / self.input_width
        y_scale = img_h / self.input_height

        boxes = []
        confidences = []
        class_ids = []

        for pred in output:
            # Scores for Red, Yellow, Green start at index 4
            scores = pred[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                # xc, yc, w, h are in 0-640 range
                xc, yc, w, h = pred[0:4]
                
                # Rescale to actual frame size
                x1 = int((xc - w / 2) * x_scale)
                y1 = int((yc - h / 2) * y_scale)
                bw = int(w * x_scale)
                bh = int(h * y_scale)
                
                boxes.append([x1, y1, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # Apply NMS to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.45)
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append((x, y, x + w, y + h, confidences[i], class_ids[i]))
        
        return detections
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple]:
        h, w = frame.shape[:2]
        
        # Preprocess
        input_img = cv2.resize(frame, (self.input_width, self.input_height))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return self.postprocess_output(output, h, w, conf_threshold)

    def draw_detections(self, image: np.ndarray, detections: List) -> np.ndarray:
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            color = self.colors.get(cls_id, (255, 255, 255))
            label = f"{self.class_names[cls_id]}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

def run_live(model_path):
    detector = TFLiteTrafficLightDetector(model_path)
    cap = cv2.VideoCapture(2)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n--- Detection Active (Press 'q' to quit) ---")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        start = time.time()
        detections = detector.detect(frame, conf_threshold=0.5)
        fps = 1.0 / (time.time() - start)
        
        # 1. PRETTY PRINT TO TERMINAL
        if detections:
            print(f"FPS: {fps:.1f} | Found: {len(detections)}")
            for d in detections:
                label = detector.class_names[d[5]]
                print(f"  - {label:7} | Conf: {d[4]:.2f} | Box: ({d[0]},{d[1]})-({d[2]},{d[3]})")
        
        # 2. DRAW ON FRAME
        frame = detector.draw_detections(frame, detections)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Traffic Light Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure this path matches your exported file
    MODEL_PATH = 'traffic_light_model.tflite'
    run_live(MODEL_PATH)