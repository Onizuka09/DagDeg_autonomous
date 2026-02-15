import rclpy
from rclpy.node import Node
from dagdeg_interfaces.msg import NavCommand
import termios
import sys
import tty

class NavCommandPublisher(Node):
    def __init__(self):
        super().__init__('command_publisher')
        
        self.publisher = self.create_publisher(NavCommand, '/nav_command', 10)
        
        self.distance_step = 20.0  # 10 cm per keypress
        self.angle_step = 40.0     # 15 degrees per keypress
        
        self.get_logger().info(f'''
        Command Publisher Controls:
        W - Move forward {self.distance_step}cm
        S - Move backward {self.distance_step}cm
        A - Rotate left {self.angle_step}°
        D - Rotate right {self.angle_step}°
        Space - STOP
        Q - RESET
        X - Exit
        ''')
        
        self.run()
        
    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
        
    def run(self):
        while rclpy.ok():
            key = self.get_key().lower()
            msg = NavCommand()
            
            if key == 'w':  # Move forward
                msg.state = 1  # MOVE
                msg.cmd = self.distance_step
                
            elif key == 's':  # Move backward
                msg.state = 1  # MOVE
                msg.cmd = -self.distance_step
                
            elif key == 'a':  # Rotate left
                msg.state = 2  # ROTATE
                msg.cmd = self.angle_step
                
            elif key == 'd':  # Rotate right
                msg.state = 2  # ROTATE
                msg.cmd = -self.angle_step
                
            elif key == ' ':  # STOP
                msg.state = 3  # STOP
                msg.cmd = 0.0
                
            elif key == 'q':  # Exit
                # Send stop before exiting
                stop_msg = NavCommand()
                stop_msg.state = 0
                stop_msg.cmd = 0.0
                self.publisher.publish(stop_msg)
            elif key == 'x':  # Exit
                # Send stop before exiting
                stop_msg = NavCommand()
                stop_msg.state = 0
                stop_msg.cmd = 0.0
                self.publisher.publish(stop_msg)
                break 
            else:
                continue
                
            self.publisher.publish(msg)
            self.get_logger().info(f'Published: state={msg.state}, cmd={msg.cmd}')
def main(args=None):
    rclpy.init(args=args)

    try:
        node = NavCommandPublisher()
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
