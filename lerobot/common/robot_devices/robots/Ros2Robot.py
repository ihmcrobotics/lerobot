import rclpy
from rclpy.node import Node

from std_msgs.msg import String

class Ros2Robot(Node):

    def __init__(self):
        super().__init__('Ros2Robot')
        self.publisher_ = self.create_publisher(String, 'robot_output', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
    def timer_callback(self):
        msg = String()
        msg.data = self.robot_type
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' %msg.data)
        self.i += 1
    def connect(self):
        pass
    def run_calibration(self):
        pass
    def teleop_step(self, record_data=False):
        pass
    def send_action(self, action):
        pass
    def disconnect(self):
        pass