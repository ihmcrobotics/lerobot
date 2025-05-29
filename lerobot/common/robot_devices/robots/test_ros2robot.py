# test_ros2robot.py

import rclpy
import pytest
import torch
import numpy as np
import threading
from unittest.mock import MagicMock

from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import JointState, Image
from lerobot.common.robot_devices.robots.configs import Ros2RobotConfig
from Ros2Robot import Ros2Robot


@pytest.fixture(scope="module", autouse=True)
def init_ros():
    rclpy.init(args=None)
    yield
    rclpy.shutdown()

def simulate_connect(robot: Ros2Robot, message: str = "connect"):
    command_pub = robot.create_publisher(String, '/lerobot/command', 10)
    msg = String()
    msg.data = message
    threading.Thread(target=lambda: rclpy.spin_once(robot, timeout_sec=1.0)).start()
    command_pub.publish(msg)
    robot.connect()


class DummyPublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


def test_send_action_not_connected():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)

    robot.action_pub = DummyPublisher()
    robot.send_action(torch.tensor([0.1, 0.2]))
    assert robot.action_pub.published == []


def test_send_action_connected():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)

    robot.action_pub = DummyPublisher()

    simulate_connect(robot)

    robot._count = 0

    t = torch.tensor([1.5, -2.5])
    robot.send_action(t)
    assert len(robot.action_pub.published) == 1
    msg = robot.action_pub.published[0]
    assert isinstance(msg, Float32MultiArray)
    assert list(msg.data) == [1.5, -2.5]
    assert robot._count == 1


def test_send_action_connected_14_joints():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)

    robot.action_pub = DummyPublisher()
    simulate_connect(robot)
    robot._count = 0

    t = torch.arange(14, dtype=torch.float32)
    robot.send_action(t)
    assert len(robot.action_pub.published) == 1
    msg = robot.action_pub.published[0]
    assert isinstance(msg, Float32MultiArray)
    assert list(msg.data) == list(np.arange(14, dtype=np.float32))
    assert robot._count == 1


def test_image_callback_converts_to_cv2():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)
    robot.action_pub = DummyPublisher()
    simulate_connect(robot)
    robot._count = 0
    cv_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    robot._bridge = MagicMock()
    robot._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding: cv_img
    ros_img = Image()
    robot._image_callback(ros_img)
    assert isinstance(robot.latest_image, np.ndarray)
    assert robot.latest_image.shape == (64, 64, 3)
    assert np.array_equal(robot.latest_image, cv_img)


def test_capture_observation_multiple_images():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)
    robot.action_pub = DummyPublisher()
    simulate_connect(robot)
    robot._count = 0
    # Set joints to 14 values
    js = JointState()
    js.position = [float(i) for i in range(14)]
    robot.latest_joints = js

    # First image observation
    dummy_img1 = np.random.randint(0, 255, (3, 640, 480), dtype=np.uint8)
    robot.latest_image = dummy_img1
    angles1, image1 = robot.capture_observation()
    assert angles1 == [float(i) for i in range(14)]
    assert isinstance(image1, np.ndarray)
    assert image1.shape == (3, 640, 480)
    assert np.array_equal(image1, dummy_img1)

    # Second image observation
    dummy_img2 = np.random.randint(0, 255, (3, 640, 480), dtype=np.uint8)
    robot.latest_image = dummy_img2
    angles2, image2 = robot.capture_observation()
    assert angles2 == [float(i) for i in range(14)]
    assert image2.shape == (3, 640, 480)
    assert np.array_equal(image2, dummy_img2)
