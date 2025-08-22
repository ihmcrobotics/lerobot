import argparse
import os.path
import threading
import time
from copy import copy
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import torch
from torch import Tensor
from torch import autocast, inference_mode
from contextlib import nullcontext

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from lerobot.robots.ihmc_ros_robot.config_ihmc_ros_robot import Ros2RobotConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def predict_action(
        observation: Dict[str, Tensor],
        policy: DiffusionPolicy,
        device: torch.device,
        use_amp: bool
) -> Tensor:
    """
    Copy and preprocess observation, then run policy.select_action under AMP/inference_mode and return the action tensor.
    Supports image tensors in HWC or CHW format.
    """
    obs = copy(observation)
    amp_ctx = (
        autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else nullcontext()
    )
    with inference_mode(), amp_ctx:
        for name, tensor in list(obs.items()):
            if tensor is None:
                continue
            # Normalize images
            if "images" in name:
                tensor = tensor.type(torch.float32) / 255.0
                if tensor.ndim == 3 and tensor.shape[2] in (1, 3):
                    tensor = tensor.permute(2, 0, 1).contiguous()
            obs[name] = tensor.unsqueeze(0).to(device)
        action = policy.select_action(obs)
        action = action.squeeze(0).cpu()
    return action


class Ros2Robot(Node):
    def __init__(self, config: Ros2RobotConfig) -> None:
        """
        Initialize ROS2 node: set up publishers, subscribers, state, and rate.
        """
        super().__init__('ihmc_ros_robot')
        self.config = config
        self.bridge = CvBridge()
        self._init_state()
        self._init_publishers()
        self._init_subscribers()
        self.rate = self.create_rate(self.config.control_frequency, self.get_clock())

    def _init_state(self) -> None:
        """Initialize all internal state variables."""
        self.command: Optional[String] = None
        self.policy_status: str = ""
        self.policy: Optional[DiffusionPolicy] = None
        self.python_status: str = ""
        self.state_hand_poses: Optional[Float32MultiArray] = None
        self.zed_left_color: Optional[np.ndarray] = None
        self.zed_right_color: Optional[np.ndarray] = None
        self.diffusion_started: bool = False
        self._logged_not_started: bool = False

    def _init_publishers(self) -> None:
        """
        Create publishers based on config.publishers.
        """
        for topic, (msg_type, qos) in self.config.publishers.items():
            name = topic.strip('/').replace('/', '_')
            pub = self.create_publisher(msg_type, topic, qos)
            setattr(self, f'{name}_pub', pub)

    def _init_subscribers(self) -> None:
        """
        Create subscribers based on config.subscribers and bind callbacks.
        """
        for topic, (msg_type, cb_name, qos) in self.config.subscribers.items():
            callback = getattr(self, cb_name)
            sub = self.create_subscription(msg_type, topic, callback, qos)
            setattr(self, f"{topic.strip('/').replace('/', '_')}_sub", sub)

    def send_action(self, action: Tensor) -> None:
        """
        Publish the action tensor as a Float32MultiArray.
        """
        arr = action.detach().cpu().numpy().astype(np.float32).flatten().tolist()
        msg = Float32MultiArray(data=arr)
        self.lerobot_lerobot_action_pub.publish(msg)

    def run_diffusion_policy(self, max_steps: int = 2000) -> None:
        """
        Main control loop: wait for observations, then repeatedly predict and send actions.
        """
        self.diffusion_started = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.python_status = "Start"

        assert self.policy is not None, "Policy must be initialized before running."
        self.policy.to(device)
        self.policy.reset()
        self.policy.eval()

        self.get_logger().info(f"Starting diffusion loop (≤ {max_steps} steps)")
        # Wait for first full observation
        while not self._observations_ready():
            self.rate.sleep()

        for step in range(max_steps):
            # Publish status
            status_msg = String(data=self.python_status)
            self.lerobot_status_pub.publish(status_msg)

            # Capture and preprocess
            angles, left_img, right_img = self.capture_observation()
            obs = {
                "observation.state": torch.tensor(angles, dtype=torch.float32, device=device),
                "observation.images.cam_zed_left": torch.tensor(left_img, dtype=torch.float32, device=device),
                "observation.images.cam_zed_right": torch.tensor(right_img, dtype=torch.float32, device=device),
            }

            # Predict and send action using unified predict_action
            action = predict_action(obs, self.policy, device, use_amp=True)
            self.send_action(action)

            if step == 10:
                self.python_status = "Diffusion"

            if step % 50 == 0:
                self.get_logger().info(f"Step {step}/{max_steps}")

            self.rate.sleep()

        # Clean up
        self.get_logger().info("Diffusion run complete.")
        self.python_status = "Done"
        self.lerobot_status_pub.publish(String(data=self.python_status))
        self.disconnect()

    def _observations_ready(self) -> bool:
        """Return True if both joint poses and both images are available."""
        return (
                self.state_hand_poses is not None
                and self.zed_left_color is not None
                and self.zed_right_color is not None
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def state_hand_poses_callback(self, msg: Float32MultiArray) -> None:
        self.state_hand_poses = msg

    def command_callback(self, msg: String) -> None:
        """
        On 'diffusion', kick off the diffusion thread.
        """
        self.command = msg
        if self.diffusion_started:
            return

        if not msg.data:
            if not self._logged_not_started:
                self.get_logger().info("Waiting for 'diffusion' command…")
                self._logged_not_started = True
            return

        if msg.data == "diffusion":
            self.get_logger().info("Received 'diffusion' command, launching policy.")
            threading.Thread(target=self.run_diffusion_policy, daemon=True).start()
        else:
            self.get_logger().info(f"Ignoring unknown command: '{msg.data}'")

    def left_color_callback(self, msg: Image) -> None:
        if self.diffusion_started:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.zed_left_color = np.transpose(cv_img, (2, 0, 1))

    def right_color_callback(self, msg: Image) -> None:
        if self.diffusion_started:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.zed_right_color = np.transpose(cv_img, (2, 0, 1))

    def status_callback(self, msg: String) -> None:
        self.policy_status = msg.data

    def capture_observation(self) -> Tuple[List[float], np.ndarray, np.ndarray]:
        """
        Returns: joint angles list, left and right image arrays (C×H×W).
        """
        angles = list(self.state_hand_poses.data)
        return angles, self.zed_left_color, self.zed_right_color

    def disconnect(self) -> None:
        """
        Publish shutdown status, destroy node, and shutdown rclpy.
        """
        self.get_logger().info("Shutting down ROS2 node.")
        self.lerobot_status_pub.publish(String(data="False"))
        time.sleep(1.0)
        self.destroy_node()
        rclpy.shutdown()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the IHMC ROS2 robot node"
    )
    parser.add_argument(
        "--trained_policy",
        type=str,
        default="H2Ozone/Circles2",
        help="Path to trained policy file (default: H2Ozone/Circles2)"
    )
    args = parser.parse_args()

    ini_path = os.path.expanduser("~/.ihmc/IHMCNetworkParameters.ini")
    with open(ini_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('RTPSDomainID='):
                value = line.split('=', 1)[1]
                ros_domain_id = int(value)
    print(f"Using RTPSDomainID from ~/.ihmc/IHMCNetworkParameters.ini: {ros_domain_id}")

    rclpy.init(args=None, domain_id=ros_domain_id)
    context = rclpy.get_default_context()

    config = Ros2RobotConfig()
    robot = Ros2Robot(config)
    robot.get_logger().info(f"ROS domain ID: {context.get_domain_id()} \nLoading trained policy file: {args.trained_policy}")

    policy_path = args.trained_policy
    robot.policy = DiffusionPolicy.from_pretrained(
        str(policy_path),
        local_files_only=False,
    )
    robot.get_logger().info("Loaded policy.")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(robot)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    spin_thread.join()


if __name__ == "__main__":
    main()
