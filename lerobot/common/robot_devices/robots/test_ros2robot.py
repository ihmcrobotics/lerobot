# test_ros2robot.py

import rclpy
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState, Image
from lerobot.common.robot_devices.robots.configs import Ros2RobotConfig
from Ros2Robot import Ros2Robot


@pytest.fixture(scope="module", autouse=True)
def init_ros():
    rclpy.init(args=None)
    yield
    rclpy.shutdown()


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
    robot.connect()
    robot._count = 0

    t = torch.tensor([1.5, -2.5])
    robot.send_action(t)
    assert len(robot.action_pub.published) == 1
    msg = robot.action_pub.published[0]
    assert isinstance(msg, Float32MultiArray)
    assert list(msg.data) == [1.5, -2.5]

    assert robot._count == 1


def test_on_timer_publishes_zero_action():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)

    robot.connect()
    robot._count = 0
    js = JointState()
    js.position = [0.0, 0.0, 0.0]
    robot.latest_joints = js

    robot.action_pub = DummyPublisher()

    robot._on_timer()

    assert len(robot.action_pub.published) == 1
    msg = robot.action_pub.published[0]
    assert list(msg.data) == [0.0, 0.0, 0.0]
    assert robot._count == 1


def test_image_callback_converts_to_cv2():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)
    cv_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    robot._bridge = MagicMock()
    robot._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding: cv_img
    ros_img = Image()
    robot._image_callback(ros_img)
    assert isinstance(robot.latest_image, np.ndarray)
    assert robot.latest_image.shape == (64, 64, 3)
    assert np.array_equal(robot.latest_image, cv_img)

def test_capture_observation():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)

    js = JointState()
    js.position = [1.0, 2.0, 3.0]
    robot.latest_joints = js

    dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    robot.latest_image = dummy_img

    angles, image = robot.capture_observation()

    assert angles == [1.0, 2.0, 3.0]
    assert isinstance(image, np.ndarray)
    assert image.shape == (64, 64, 3)
    assert np.array_equal(image, dummy_img)
