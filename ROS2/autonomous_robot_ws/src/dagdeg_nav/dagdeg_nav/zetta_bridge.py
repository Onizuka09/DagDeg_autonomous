import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
import math
from .zetta_protocol import ZettaProtocol, ZettaPacketType, create_struct_builder
import struct  # Don't forget this!
from dagdeg_interfaces.msg import NavCommand

NAV_COMNAD = {"RESET": 0, "POS": 1, "ANG": 2, "STOP": 3}



class ZettaBridgeNode(Node):
    def __init__(self):
        super().__init__("zetta_bridge")

        # ROS 2 Publishers
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Initialize Zetta for your actual struct
        self.zetta = ZettaProtocol(port="/dev/ttyACM0", baudrate=115200)

        # Register parser for YOUR OdometryTypedef (12 floats + 2 shorts = 52 bytes)
        def parse_odometry_struct(payload):
            """Parse your exact OdometryTypedef structure"""
            # < = little-endian, 12f = 12 floats, 2h = 2 signed shorts
            values = struct.unpack("<6f2h2f", payload)

            return {
                "x": values[0],  # cm
                "y": values[1],  # cm
                "theta": values[2],  # radians
                "v": values[4],  # cm/s
                "w": values[5],  # rad/s
                "left_encoder": values[6],  # ticks
                "right_encoder": values[7],  # ticks
                "total_distance": values[8],  # cm
            }

        self.odo_dict = {
            "x": float(0),  # cm
            "y": float(0),  # cm
            "theta": float(0),  # radians
            "v": float(0),  # cm/s
            "w": float(0),  # rad/s
            "left_encoder": float(0),  # ticks
            "right_encoder": float(0),  # ticks
            "total_distance": float(0),  # cm
        }

        self.zetta.register_packet_handler(
            ZettaPacketType.MSG_PUBLISH, parser=parse_odometry_struct, builder= create_struct_builder('<Bf')
        )
        

        self.zetta.start()

        # create subscriber for navigation commands 
        self.nav_command = self.create_subscription(NavCommand,'/nav_command',self.nav_command_clbk,10)
        # Timer for processing (100Hz)
        self.timer = self.create_timer(0.01, self.update)

        self.get_logger().info("Zetta Bridge Node started")

    def update(self):
        """Main processing loop - called every 10ms"""
        # Check for new packets (non-blocking)
        packet = self.zetta.get_packet(timeout=0)

        if packet and packet.type == ZettaPacketType.MSG_PUBLISH:
            try:
                odom_data = self.zetta.process_packet(packet)
                self.odo_dict = odom_data
                self.publish_odometry(odom_data)

                # Log at lower frequency to avoid spam
                if self.get_clock().now().nanoseconds % 1000000000 < 10000000:  # ~1Hz
                    self.get_logger().info(
                    f'Robot: x={odom_data["x"]:.1f}cm, '
                    f'y={odom_data["y"]:.1f}cm, '
                    f'θ={math.degrees(odom_data["theta"]):.1f}°'
                    )
            except Exception as e:
                self.get_logger().error(f"Error processing packet: {e}")
        elif not packet:
            self.publish_odometry(self.odo_dict)
    def nav_command_clbk(self,msg: NavCommand): 
        state = (list(NAV_COMNAD.keys())[list(NAV_COMNAD.values()).index(msg.state)])
        self.get_logger().info(f"state {state}, value: {msg.cmd}")
        if msg.cmd is None : 
            msg.cmd = 0.0 
        msg.state = msg.state & 0x0ff  
        data = struct.pack('<Bf',msg.state , msg.cmd )
        data = (msg.state , msg.cmd)
        if self.zetta.send(ZettaPacketType.MSG_PUBLISH , data ) :
            self.get_logger().info("transmitted packet succ")
        else : 
            self.get_logger().error("Error: transmitted packet ")
    def publish_odometry(self, data):
        """Publish odometry data to ROS 2"""
        now = self.get_clock().now().to_msg()

        # 1. Publish TF transform (odom -> base_link)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "odom"  # Parent frame
        t.child_frame_id = "base_link"  # Child frame (robot)
        t.transform.translation.x = data["x"] / 100.0  # cm -> meters
        t.transform.translation.y = data["y"] / 100.0
        t.transform.translation.z = 0.0
        t.transform.rotation = self.yaw_to_quaternion(data["theta"])
        self.tf_broadcaster.sendTransform(t)

        # 2. Publish Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # Position
        odom_msg.pose.pose.position.x = data["x"] / 100.0
        odom_msg.pose.pose.position.y = data["y"] / 100.0
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = t.transform.rotation

        # Velocity (twist)
        odom_msg.twist.twist.linear.x = data["v"] / 100.0  # cm/s -> m/s
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = data["w"]

        # Add some covariance (important for navigation)
        # Position covariance (0.1 meter uncertainty)
        odom_msg.pose.covariance[0] = 0.1  # x
        odom_msg.pose.covariance[7] = 0.1  # y
        odom_msg.pose.covariance[35] = 0.1  # yaw

        # Velocity covariance
        odom_msg.twist.covariance[0] = 0.1  # vx
        odom_msg.twist.covariance[35] = 0.1  # wz

        self.odom_pub.publish(odom_msg)

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle (radians) to ROS quaternion"""
        from geometry_msgs.msg import Quaternion

        return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def main(args=None):
    rclpy.init(args=args)

    try:
        node = ZettaBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        # Cleanup
        node.zetta.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
