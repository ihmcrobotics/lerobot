import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import threading
import time
class ConnectPublisher(Node):
    def __init__(self):
        super().__init__('connect_publisher')
        self.publisher_ = self.create_publisher(String, '/lerobot/connect', 10)
        self.diffus_pub = self.create_publisher(String, '/lerobot/command', 10)
        self.pubPoses = self.create_publisher(Float32MultiArray, '/lerobot/state/hand_poses', 10)
        self.zedLeft = self.create_publisher(Image, '/zed/left/color', 10)
        self.zedRight = self.create_publisher(Image, '/zed/right/color', 10)

        self.status_sub = self.create_subscription(String, '/lerobot/status', self._lerobot_status, 10)
        self.pose_sub = self.create_subscription(Float32MultiArray, '/lerobot/action/hand_poses', self._lerobot_pose, 10)
        msg = String()
        msg.data = "test"
        self.publisher_.publish(msg)
        self.get_logger().info('Published connect command')
        self.status = "True"
        self.current_pose = None

    def _lerobot_pose(self, msg):
        self.current_pose = np.array(msg.data, dtype=np.float32)
        print(self.current_pose)
        # self.get_logger().info("Received pose")
    def _lerobot_status(self, msg):
        self.status = msg.data
        self.get_logger().info(self.status)

    def diffusion(self):
        bridge = CvBridge()
        msg1 = String()
        msg1.data = "diffusion"
        self.diffus_pub.publish(msg1)
        time.sleep(5)
        # Continue streaming data
        while self.status == "True":
            cv_img = np.random.randint(0, 255, (3, 640, 480), dtype=np.uint8)
            ros_img_left = bridge.cv2_to_imgmsg(np.transpose(cv_img, (1, 2, 0)), encoding='bgr8')
            ros_img_right = bridge.cv2_to_imgmsg(np.transpose(cv_img, (1, 2, 0)), encoding='bgr8')
            self.zedLeft.publish(ros_img_left)
            self.zedRight.publish(ros_img_right)
            js = Float32MultiArray()
            if self.current_pose is not None:
                js.data = self.current_pose.tolist()
            else:
                js.data = [float(np.random.randint(0,200)) for _ in range(14)]
            self.pubPoses.publish(js)
            # self.get_logger().info('Published live streaming data')
            time.sleep(2)


def main():
    rclpy.init()
    connect_publisher = ConnectPublisher()
    spin_thread = threading.Thread(target=rclpy.spin, args=(connect_publisher,))
    spin_thread.start()
    connect_publisher.diffusion()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
