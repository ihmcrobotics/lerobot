import rclpy
from rclpy.node import Node

import torch
import numpy as np

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

from typing import Optional, Tuple, List
import numpy as np

from lerobot.common.robot_devices.robots.configs import Ros2RobotConfig


class Ros2Robot(Node):
    def __init__(self, config: Ros2RobotConfig):
        super().__init__('ros2robot')
        self.config = config

        self.action_pub = self.create_publisher(
            Float32MultiArray,
            config.action_topic,
            10
        )

        # --- Subscriber for joint states ---
        self.joint_state_sub = self.create_subscription(
            JointState,
            config.joint_state_topic,
            self._joint_state_callback,
            10
        )

        if config.image_topic is not None:
            self.image_sub = self.create_subscription(
                Image,
                config.image_topic,
                self._image_callback,
                10
            )
            self._bridge = CvBridge()

        period = 1.0 / config.control_frequency
        self.timer = self.create_timer(period, self._on_timer)
        self.latest_joints = None
        self.latest_image = None
        self._count = 0
        self.is_connected = False

    def connect(self):
        self.get_logger().info('Connecting to robot...')
        self.is_connected = True
        self.get_logger().info('Connected.')

    def send_action(self, action: torch.Tensor):
        if not self.is_connected:
            self.get_logger().warning('Not connected: dropping action')
            return
        arr = action.detach().cpu().numpy().astype(np.float32)
        msg = Float32MultiArray(data=arr.flatten().tolist())
        self.action_pub.publish(msg)
        self.get_logger().debug(f'Published action #{self._count}: {arr.tolist()}')
        self._count += 1

    def _joint_state_callback(self, msg: JointState):
        self.latest_joints = msg
        self.get_logger().debug(f'Received joints: {msg.position}')

    def _image_callback(self, msg: Image):
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_image = cv_img
        self.get_logger().debug('Received new image frame')

    def _on_timer(self):
        if self.latest_joints is None:
            return
        desired = torch.zeros(len(self.latest_joints.position))
        self.send_action(desired)

    def capture_observation(self) -> Tuple[List[float], Optional[np.ndarray]]:
        if hasattr(self, 'latest_joints') and self.latest_joints is not None:
            angles: List[float] = list(self.latest_joints.position)
        else:
            angles = []
        image: Optional[np.ndarray] = getattr(self, 'latest_image', None)

        return angles, image

    def disconnect(self):
        """
        Clean up before shutdown.
        """
        self.get_logger().info('Shutting down...')
        self.destroy_node()
        rclpy.shutdown()



