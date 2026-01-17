import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
import cv2 
class CameraRawNode(Node):
    def __init__(self,):
        super().__init__('camera_raw_node')

        self.publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            10)
         
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(2,cv2.CAP_V4L2)  # 0 = default camera
        if not self.cap.isOpened():
            self.get_logger().error("cannot open camera")
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)
    def timer_callback(self):
        ret,frame = self.cap.read()
        if not ret: 
            self.get_logger().warning("Failed to capture frame")
            return 
        msg = self.bridge.cv2_to_imgmsg(frame,encoding='bgr8')
        self.publisher.publish(msg)
def main(args=None): 
    rclpy.init(args=args)
    CamNode = CameraRawNode()
    rclpy.spin(CamNode)
    CamNode.destroy_node()
    rclpy.shutdown()
