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
    """Initialize ROS2 once for all tests and shut down when done."""
    rclpy.init(args=None)
    yield
    rclpy.shutdown()


class DummyPublisher:
    """A little stub to capture published messages."""
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


def test_send_action_not_connected():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)

    robot.action_pub = DummyPublisher()
    # without connect(), send_action should drop
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

    # compare list(msg.data), not the raw array('f', ...)
    assert len(robot.action_pub.published) == 1
    msg = robot.action_pub.published[0]
    assert isinstance(msg, Float32MultiArray)
    assert list(msg.data) == [1.5, -2.5]

    # counter should have incremented
    assert robot._count == 1


def test_on_timer_publishes_zero_action():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)

    robot.connect()
    robot._count = 0

    # pretend we've received joint states
    js = JointState()
    js.position = [0.0, 0.0, 0.0]
    robot.latest_joints = js

    robot.action_pub = DummyPublisher()

    # call the periodic callback
    robot._on_timer()

    assert len(robot.action_pub.published) == 1
    msg = robot.action_pub.published[0]
    # again, wrap in list()
    assert list(msg.data) == [0.0, 0.0, 0.0]
    assert robot._count == 1


def test_image_callback_converts_to_cv2():
    config = Ros2RobotConfig(mock=True)
    robot = Ros2Robot(config)

    # Create a dummy OpenCV image
    cv_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Stub out the bridge on this instance only
    robot._bridge = MagicMock()
    robot._bridge.imgmsg_to_cv2 = lambda msg, desired_encoding: cv_img

    # Create a dummy ROS Image (its contents don't matter since we stub imgmsg_to_cv2)
    ros_img = Image()

    # Invoke the callback
    robot._image_callback(ros_img)

    # Now latest_image should be your numpy array
    assert isinstance(robot.latest_image, np.ndarray)
    assert robot.latest_image.shape == (64, 64, 3)
    assert np.array_equal(robot.latest_image, cv_img)
