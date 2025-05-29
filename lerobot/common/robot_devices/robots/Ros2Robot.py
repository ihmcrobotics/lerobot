import time

import rclpy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from rclpy.node import Node

import torch
import numpy as np
import threading
import gymnasium as gym
import gym_pusht
import imageio

from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

from typing import Optional, Tuple, List
from pathlib import Path

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
        self.command = ""
        self.connect_event = threading.Event()
        self.policy_status = ""
        self.latest_joints = None  # type: Optional[JointState]
        self.latest_image_right = None   # type: Optional[np.ndarray]
        self.latest_image_left = None
        self._count = 0            # Counter for action messages
        self.is_connected = False  # Connection status flag

    def connect(self):
        """
        Mark the robot as connected and enable sending actions.
        Logs connection status to the ROS2 console.
        """
        self.get_logger().info('Waiting for /lerobot/command to connect...')
        self.connect_event.wait()  # Blocks here until _command_callback sets the event
        self.get_logger().info('Command received. Connecting to robot...')
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
        self.lerobot_action_hand_poses_pub.publish(msg)
        self.get_logger().info(f'Published action #{self._count}: {arr.tolist()}')
        self._count += 1
    #TODO: Needs to be tested with our already trained thing
    def run_diffusion_policy(self, max_steps=300, policy_path=Path("outputs/train/pretrained_model")):
        """
        Runs the diffusion policy on the real robot through ROS2 topics,

        Args:
            max_steps (int): Number of steps to run.
            policy_path (str): HF repo ID or local path for the pretrained policy.
        """
        import torch
        from pathlib import Path
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        # 1. Load policy and select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = DiffusionPolicy.from_pretrained(policy_path)
        policy.to(device)
        policy.reset()

        self.get_logger().info(f"Running diffusion policy for up to {max_steps} steps...")

        step = 0
        while step < max_steps and self.is_connected:
            # 2. Capture observation from ROS
            joint_angles, image_left, image_right = self.capture_observation()
            # Wait until both are available
            if not joint_angles or image_left is None or image_right is None:
                rclpy.spin_once(self, timeout_sec=0.05)
                continue

            # 3. Prepare tensors for policy (match input names and formats)
            state = torch.tensor(joint_angles, dtype=torch.float32, device=device).unsqueeze(0)
            img_tensor_left = torch.tensor(image_left, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            img_tensor_right = torch.tensor(image_right, dtype=torch.float32, device=device).unsqueeze(0) / 255.0

            # Compose policy input
            obs = {
                "observation.state": state,
                "observation.images.cam_zed_left": img_tensor_left,
                "observation.images.cam_zed_right": img_tensor_right,
            }

            # 4. Inference
            with torch.no_grad():
                action = policy.select_action(obs)
                action = action.squeeze(0)

            # 5. Send action to robot
            self.send_action(action)

            # 6. Optional: Logging
            self.get_logger().info(f"Step {step}: Action sent.")

            # 7. Wait for next observation
            rclpy.spin_once(self, timeout_sec=0.05)
            step += 1

        self.get_logger().info("Diffusion policy run complete or robot disconnected.")

    def _joint_state_callback(self, msg: JointState):
        """
        Callback invoked when a new JointState message arrives.
        Stores the latest joint positions for later retrieval.
        """
        self.latest_joints = msg
        self.get_logger().info(f'Received joints: {msg.position}')

    def _connect_callback(self, msg: String):
        """
        Callback for the /lerobot/connect topic.
        Triggers the connection process when a message is received.
        """
        self.get_logger().info(f'Received connect: {msg.data}')
        self.connect_event.set()

    def _command_callback(self, msg: String):
        self.command = msg
        self.get_logger().info(f'Received command: {msg.data}')
        if self.command.data == '':
            return
        elif self.command.data == 'diffusion':
            self.run_diffusion_policy()

    def _image_left_callback(self, msg: Image):
        """
        Callback invoked when a new Image message arrives.
        Converts the ROS Image message to an OpenCV image and stores it.
        """
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_image_left = cv_img
        self.get_logger().info('Received new image frame')
    def _image_right_callback(self, msg: Image):
        """
        Callback invoked when a new Image message arrives.
        Converts the ROS Image message to an OpenCV image and stores it.
        """
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_image_right = cv_img
        self.get_logger().info('Received new image frame')

    def _status_callback(self, msg: String):
        """
        Callback invoked when a new status message arrives.
        Sets the policy type to recieved message
        """
        self.policy_status = msg
        self.get_logger().info(f'Received status message: {msg}')

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

    def capture_observation(self) -> Tuple[List[float], Optional[np.ndarray], Optional[np.ndarray]]:
        # Extract joint positions if available
        if hasattr(self, 'latest_joints') and self.latest_joints is not None:
            angles: List[float] = list(self.latest_joints.position)
        else:
            angles = [np.random.randint(0,200) for _ in range(14)]
        # Extract the latest image frame if available
        image_left: Optional[np.ndarray] = getattr(self, 'latest_image_left', None)
        if image_left is not None and image_left.ndim == 3:
            # Transpose from HWC to CHW
            image_left = np.transpose(image_left, (2, 0, 1))
        else:
            image_left = np.random.randint(0, 255, (3, 640, 480), dtype=np.uint8)
        image_right: Optional[np.ndarray] = getattr(self, 'latest_image_right', None)
        if image_right is not None and image_right.ndim == 3:
            # Transpose from HWC to CHW
            image_right = np.transpose(image_right, (2, 0, 1))
        else:
            image_right = np.random.randint(0, 255, (3, 640, 480), dtype=np.uint8)
        return angles, image_left, image_right

    def disconnect(self):
        """
        Cleanly shut down the ROS2 node and stop all processing.
        Logs the shutdown and destroys the node.
        """
        self.get_logger().info('Shutting down...')
        self.is_connected = False
        self.destroy_node()
        rclpy.shutdown()