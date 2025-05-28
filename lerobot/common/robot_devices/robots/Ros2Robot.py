import rclpy
from rclpy.node import Node

import torch
import numpy as np

from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

from typing import Optional, Tuple, List

from lerobot.common.robot_devices.robots.configs import Ros2RobotConfig


class Ros2Robot(Node):
    def __init__(self, config: Ros2RobotConfig):
        """
        Initialize the ROS2 robot node.
        Sets up publishers for actions, subscribers for joint states and images,
        a control timer, and state variables to track the latest observations.
        """
        super().__init__('ros2robot')
        self.config = config

        # Publisher for sending actions to the robot
        for topic, (msg_type, qos) in config.publishers.items():
            attr = topic.strip('/').replace('/', '_') + '_pub'
            setattr(self, attr, self.create_publisher(msg_type, topic, qos))
        self._bridge = None
        for topic, (msg_type, cb_name, qos) in config.subscribers.items():
            callback = getattr(self, cb_name)
            attr = topic.strip('/').replace('/','_') + '_sub'
            setattr(self, attr, self.create_subscription(msg_type, topic, callback, qos))
            if msg_type is Image and self._bridge is None:
                self._bridge = CvBridge()

        # Timer for periodic control callbacks at the configured frequency
        period = 1.0 / config.control_frequency
        self.timer = self.create_timer(period, self._on_timer)

        # Initialize state variables
        self.latest_joints = None  # type: Optional[JointState]
        self.latest_image = None   # type: Optional[np.ndarray]
        self._count = 0            # Counter for action messages
        self.is_connected = False  # Connection status flag

    def connect(self):
        """
        Mark the robot as connected and enable sending actions.
        Logs connection status to the ROS2 console.
        """
        self.get_logger().info('Connecting to robot...')
        self.is_connected = True
        self.get_logger().info('Connected.')

    def send_action(self, action: torch.Tensor):
        """
        Publish the provided action tensor to the robot.
        If not connected, logs a warning and drops the message.
        Converts the tensor to a Float32MultiArray message before publishing.
        """
        if not self.is_connected:
            self.get_logger().warning('Not connected: dropping action')
            return
        # Convert action tensor to numpy and then to ROS message
        arr = action.detach().cpu().numpy().astype(np.float32)
        msg = Float32MultiArray(data=arr.flatten().tolist())
        self.action_pub.publish(msg)
        self.get_logger().debug(f'Published action #{self._count}: {arr.tolist()}')
        self._count += 1

    def _joint_state_callback(self, msg: JointState):
        """
        Callback invoked when a new JointState message arrives.
        Stores the latest joint positions for later retrieval.
        """
        self.latest_joints = msg
        self.get_logger().debug(f'Received joints: {msg.position}')

    def _command_callback(self, msg: String):
        #TODO: Figure out how to make this have connect wait on it
        pass

    def _image_callback(self, msg: Image):
        """
        Callback invoked when a new Image message arrives.
        Converts the ROS Image message to an OpenCV image and stores it.
        """
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_image = cv_img
        self.get_logger().debug('Received new image frame')

    def _on_timer(self):
        """
        Timer callback running at the control frequency.
        If joint states are available, sends a zero-action command.
        """
        # Do nothing until at least one joint state is received
        if self.latest_joints is None:
            return
        # Create a zero-action vector matching the number of joints
        desired = torch.zeros(len(self.latest_joints.position))
        self.send_action(desired)

    def capture_observation(self) -> Tuple[List[float], Optional[np.ndarray]]:
        """
        Retrieve the latest observations from the robot.
        Returns:
            angles: List of the most recent joint positions (floats).
            image: Most recent camera frame as an OpenCV array, or None if not available.
        """
        # Extract joint positions if available
        if hasattr(self, 'latest_joints') and self.latest_joints is not None:
            angles: List[float] = list(self.latest_joints.position)
        else:
            angles = []
        # Extract the latest image frame if available
        image: Optional[np.ndarray] = getattr(self, 'latest_image', None)

        return angles, image

    def disconnect(self):
        """
        Cleanly shut down the ROS2 node and stop all processing.
        Logs the shutdown and destroys the node.
        """
        self.get_logger().info('Shutting down...')
        self.is_connected = False
        self.destroy_node()
        rclpy.shutdown()
