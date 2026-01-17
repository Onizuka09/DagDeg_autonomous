import rclpy 
import os 
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from vision_msgs.msg import (
    Detection2D, 
    Detection2DArray, 
    ObjectHypothesisWithPose, 
    ObjectHypothesis, # Added this for hypothesis assignment
    Pose2D, 
    Point2D
)
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

# Import the detector module (ensure it's in your PYTHONPATH)
from vision_perception.traffic_ligh_tflite_module import TFLiteTrafficLightDetector 

class TrafficLightDetectorNode(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')
        
        # 1. Define Class Names (Crucial for labels)
        
        self.class_names = ['Green', 'Red', 'Yellow'] 

        # 2. Setup Paths and Model
        model_path = os.path.join(
            get_package_share_directory('vision_perception'),
            'models',
            'traffic_light_model.tflite'
        )
        self.get_logger().info(f"Loading TFLITE model: {model_path}")
        
        # Force CPU (fixing your GPU issue)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        self.detector = TFLiteTrafficLightDetector(model_path)
        self.cvBridge = CvBridge() 

        # 3. ROS Topics
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_rcv_callback, 10)
        
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/traffic_light/detections', 10)

        self.image_bbx_pub = self.create_publisher(
            Image, '/traffic_light_node/image_annotated', 10)

        self.get_logger().info("Traffic Light Detector Node started")

    def image_rcv_callback(self, msg: Image):
        # Convert ROS2 image to CV2
        frame = self.cvBridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Perform detection (Assumes your detector returns list/array of results)
        # Note: Changed variable name from 'detections' to 'raw_results' for clarity
        raw_results = self.detector.detect(frame, conf_threshold=0.5)

        detect_arr_msg = Detection2DArray()
        detect_arr_msg.header = msg.header 

        best_conf = 0.0
        best_data = None

        # 4. Logic Fix: Iterate through your actual detector output
        # Assuming detector output format: [[x1, y1, x2, y2, score, cls_id], ...]
        for res in raw_results:
            x1, y1, x2, y2, conf, cls_id = res
            
            if conf > best_conf:
                best_conf = conf
                best_data = (x1, y1, x2, y2, int(cls_id))

        # 5. Build the ROS 2 Detection Message
        if best_data is not None: 
            x1, y1, x2, y2, cls_id = best_data
            
            det = Detection2D()
            
            # Setup Bounding Box
            det.bbox.center = Pose2D()
            det.bbox.center.position = Point2D()
            det.bbox.center.position.x = float((x1 + x2) / 2.0)
            det.bbox.center.position.y = float((y1 + y2) / 2.0)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            # Setup Hypothesis
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = str(cls_id)
            h.hypothesis.score = float(best_conf)
            
            det.results.append(h)
            detect_arr_msg.detections.append(det)

            # 6. Draw on frame
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            color = (0, 0, 255) if 'red' in label.lower() else (0, 255, 0)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {best_conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Publish detections
            self.detection_pub.publish(detect_arr_msg)

        # 7. Publish annotated image
        image_bbx_msg = self.cvBridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image_bbx_msg.header = msg.header 
        self.image_bbx_pub.publish(image_bbx_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()