import rclpy 
import os 
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose , Pose2D,  Point2D




from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

from ament_index_python.packages import get_package_share_directory


model_path = os.path.join(get_package_share_directory('vision_perception'),'models','traffic_light_model_best.pt')
model_confidence = 0.4 
# Force CPU (i have issues with my GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class TrafficLightDetectorNode(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names 
        
        self.cvBridge = CvBridge() 
        # subscribe to the image_raw node 
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_rcv_callback,
            10
        )
        # publish the predections 
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/traffic_light/detections',
            10
        )

        # publish annotated images (bbx)
        self.image_bbx_pub = self.create_publisher(
            Image,
            '/traffic_light_node/image_annotated',
            10
        )
        self.get_logger().info("Traffic Light Detector Node started")
    def image_rcv_callback(self,msg: Image):
        # convert the image from ros2 to cv2 
        frame =self.cvBridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        results = self.model(frame,conf=model_confidence,verbose=False)
        detect_arr = Detection2DArray()
        # keep the same header ( timestamp and frame_id)
        detect_arr.header = msg.header 
        best_box = None
        best_conf = 0.0
        best_cls_id = None
        best_coords = None
        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf > best_conf:
                    best_conf = conf
                    best_cls_id = cls_id
                    best_coords = (x1, y1, x2, y2)
                    best_box = box  # Store the whole box if needed

                # ---- Detection message ----
        if best_box is not None: 
            x1, y1, x2, y2 = best_coords
            conf = best_conf
            cls_id = best_cls_id

            det = Detection2D()
            
            
            center_pose = Pose2D()
            center_pose.position =  Point2D()
            center_pose.position.x = float((x1 + x2) / 2.0)
            center_pose.position.y = float((y1 + y2) / 2.0)
            center_pose.theta = 0.0

            # det.bbox.center = center_pose
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = str(cls_id)
            h.hypothesis.score = conf
            det.results.append(h)
            detect_arr.detections.append(det)
            # ---- Draw bounding box ----
            label = self.class_names[cls_id]
            color = (0, 0, 255) if 'red' in label.lower() else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # publish the dection result 
            self.detection_pub.publish(detect_arr)
            # publsih the image 
        image_bbx_msg = self.cvBridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image_bbx_msg.header = msg.header 
        self.image_bbx_pub.publish(image_bbx_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()