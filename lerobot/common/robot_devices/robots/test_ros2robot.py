# test_ros2robot.py
import rclpy
import pytest
import torch
import numpy as np
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
from Ros2Robot import Ros2Robot
from lerobot.common.robot_devices.robots.configs import Ros2RobotConfig

@pytest.fixture(scope="module", autouse=True)
def init_ros():
    rclpy.init(args=None)
    yield
    rclpy.shutdown()

def simulate_connect(robot: Ros2Robot, message: str = "connect"):
    """
    Publish a String on '/lerobot/command' and spin once
    so that robot.connect() will unblock.
    """
    cmd_pub = robot.create_publisher(String, '/lerobot/command', 10)
    rclpy.spin_once(robot, timeout_sec=0.1)
    msg = String()
    msg.data = message
    cmd_pub.publish(msg)
    rclpy.spin_once(robot, timeout_sec=0.1)
    robot.connect()

def test_send_action_not_connected():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)
    received = []
    robot.create_subscription(
        Float32MultiArray,
        '/lerobot/action/hand_poses',
        lambda msg: received.append(msg),
        10,
    )
    t = torch.arange(14, dtype=torch.float32)
    robot.send_action(t)
    rclpy.spin_once(robot, timeout_sec=0.1)

    assert received == []

def test_send_action_connected_14_joints():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)
    received = []
    robot.create_subscription(
        Float32MultiArray,
        '/lerobot/action/hand_poses',
        lambda msg: received.append(msg),
        10,
    )

    simulate_connect(robot)
    t = torch.arange(14, dtype=torch.float32)
    robot.send_action(t)
    rclpy.spin_once(robot, timeout_sec=0.1)

    assert len(received) == 1
    msg = received[0]
    assert isinstance(msg, Float32MultiArray)
    assert list(msg.data) == list(np.arange(14, dtype=np.float32))
    assert robot._count == 1

def test_image_subscription_converts_to_cv2():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)
    simulate_connect(robot)
    bridge = CvBridge()
    pub = robot.create_publisher(Image, '/zed/left/color', 10)
    cv_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    ros_img = bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
    pub.publish(ros_img)
    rclpy.spin_once(robot, timeout_sec=0.1)

    assert isinstance(robot.latest_image, np.ndarray)
    assert robot.latest_image.shape == (64, 64, 3)
    assert np.array_equal(robot.latest_image, cv_img)

def test_capture_observation_via_ros2():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)
    simulate_connect(robot)

    js_pub = robot.create_publisher(JointState, '/lerobot/state/hand_poses', 10)
    js = JointState()
    js.position = [float(i) for i in range(14)]
    js_pub.publish(js)
    rclpy.spin_once(robot, timeout_sec=0.1)

    img_pub = robot.create_publisher(Image, '/zed/left/color', 10)
    bridge = CvBridge()
    cv_img = np.random.randint(0, 255, (3, 640, 480), dtype=np.uint8)
    ros_img = bridge.cv2_to_imgmsg(
        np.transpose(cv_img, (1, 2, 0)), encoding='bgr8'
    )
    img_pub.publish(ros_img)
    rclpy.spin_once(robot, timeout_sec=0.1)

    angles, image = robot.capture_observation()
    assert angles == [float(i) for i in range(14)]
    assert isinstance(image, np.ndarray)
    assert image.shape == (3, 640, 480)
    assert np.array_equal(image, cv_img)
