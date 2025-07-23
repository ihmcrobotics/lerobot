import argparse
import threading
import time
from copy import copy

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import torch
import numpy as np

from contextlib import nullcontext

from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from typing import Optional, Tuple, List
from pathlib import Path
from lerobot.robots.ihmc_ros_robot.config_ihmc_ros_robot import Ros2RobotConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.utils import get_safe_torch_device


class Ros2Robot(Node):
    def __init__(self, config: Ros2RobotConfig):
        """
        Initialize the ROS2 robot node.
        Sets up publishers for actions, subscribers for joint states and images,
        and state variables to track the latest observations.
        """
        super().__init__('ihmc_ros_robot')
        self.config = config

        for topic, (msg_type, qos) in config.publishers.items():
            attr = topic.strip('/').replace('/', '_') + '_pub'
            setattr(self, attr, self.create_publisher(msg_type, topic, qos))
        self.bridge = None
        for topic, (msg_type, cb_name, qos) in config.subscribers.items():
            callback = getattr(self, cb_name)
            attr = topic.strip('/').replace('/', '_') + '_sub'
            setattr(self, attr, self.create_subscription(msg_type, topic, callback, qos))
            if msg_type is Image and self.bridge is None:
                self.bridge = CvBridge()

        # Initialize state variables
        self.command = ""
        self.policyStatus = ""
        self.stateHandPoses = None
        self.zedRightColor = None
        self.zedLeftColor = None
        self.diffusionStart = False
        self.diffusionPrint = False

    def send_action(self, action: torch.Tensor):
        """
        Publish the provided action tensor to the robot.
        If not connected, logs a warning and drops the message.
        Converts the tensor to a Float32MultiArray message before publishing.
        """
        # Convert action tensor to numpy and then to ROS message
        arr = action.detach().cpu().numpy().astype(np.float32)
        msg = Float32MultiArray(data=arr.flatten().tolist())
        self.lerobot_lerobot_action_hand_poses_pub.publish(msg)
        # TODO: Get rid of sleep for a throttler or something of the sort
        time.sleep(0.25)

    def run_diffusion_policy(self, max_steps=100, policy_path=Path("H2Ozone/Circles2")):
        """
        Runs the diffusion policy on the real robot through ROS2 topics,

        Args:
            max_steps (int): Number of steps to run.
            policy_path (str): HF repo ID or local path for the pretrained policy.
        """

        self.diffusionStart = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = DiffusionPolicy.from_pretrained(
            str(policy_path)
        )
        policy.to(device)
        policy.reset()
        policy.eval()

        self.get_logger().info(f"Running diffusion policy for up to {max_steps} steps...")

        step = 0
        while step < max_steps:
            if not (self.stateHandPoses and self.zedLeftColor is not None and self.zedRightColor is not None):
                continue

            state_hand_poses, left_color, right_color = self.capture_observation()
            state = torch.tensor(state_hand_poses, dtype=torch.float32, device=device)
            img_tensor_left = torch.tensor(left_color, dtype=torch.float32, device=device)
            img_tensor_right = torch.tensor(right_color, dtype=torch.float32, device=device)

            obs = {
                "observation.state": state,
                "observation.images.cam_zed_left": img_tensor_left,
                "observation.images.cam_zed_right": img_tensor_right,
            }

            action = predict_action(obs, policy, get_safe_torch_device(policy.config.device), True)

            self.send_action(action)

            self.get_logger().info(f'Published action #{step}')
            step += 1

        self.get_logger().info("Diffusion policy run complete or robot disconnected.")
        time.sleep(1)
        self.disconnect()
        time.sleep(1)

    def state_hand_poses_callback(self, msg: Float32MultiArray):
        """
        Callback invoked when a new JointState message arrives.
        Stores the latest joint positions for later retrieval.
        """
        self.stateHandPoses = msg

    def command_callback(self, msg: String):
        """
        Callback for the /lerobot/command topic.
        Switches from no policy to diffusion policy.
        """
        self.command = msg
        if self.diffusionStart:
            return
        if self.command.data == '':
            if not self.diffusionPrint:
                self.get_logger().info("Diffusion policy not started yet...")
                self.diffusionPrint = True
            return
        elif self.command.data == 'diffusion':
            # Launch diffusion in separate thread
            self.get_logger().info(f'Received diffusion command: {msg.data}')
            threading.Thread(target=self.run_diffusion_policy, daemon=True).start()
        else:
            self.get_logger().info(f'Command callback is: {msg.data}')

    def left_color_callback(self, msg: Image):
        """
        Callback invoked when a new Image message arrives.
        Converts the ROS Image message to an OpenCV image and stores it.
        """
        if not self.diffusionStart:
            return
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.zedLeftColor = cv_img

    def right_color_callback(self, msg: Image):
        """
        Callback invoked when a new Image message arrives.
        Converts the ROS Image message to an OpenCV image and stores it.
        """
        if not self.diffusionStart:
            return
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.zedRightColor = cv_img

    def status_callback(self, msg: String):
        """
        Callback invoked when a new status message arrives.
        Sets the policy type to recieved message
        """
        self.policyStatus = msg.data

    def capture_observation(self) -> Tuple[List[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if hasattr(self, 'stateHandPoses') and self.stateHandPoses is not None:
            angles: List[float] = list(self.stateHandPoses.data)
        else:
            angles = None

        image_left: Optional[np.ndarray] = self.zedLeftColor
        if image_left is not None and image_left.ndim == 3:
            image_left = np.transpose(image_left, (2, 0, 1))

        image_right: Optional[np.ndarray] = self.zedRightColor
        if image_right is not None and image_right.ndim == 3:
            image_right = np.transpose(image_right, (2, 0, 1))

        return angles, image_left, image_right

    def disconnect(self):
        """
        Cleanly shut down the ROS2 node and stop all processing.
        Logs the shutdown and destroys the node.
        """
        self.get_logger().info('Shutting down...')
        status = String()
        status.data = "False"
        self.lerobot_status_pub.publish(status)
        time.sleep(2)
        self.destroy_node()
        time.sleep(1)
        rclpy.shutdown()


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(), torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)
        action = policy.select_action(observation)
        action = action.squeeze(0)
        action = action.to("cpu")
    return action


def main():
    # TODO: Figure out way to look at IHMCNetwork to set the domain id correctly
    parser = argparse.ArgumentParser(description="Launch the IHMC ROS2 robot node")
    parser.add_argument(
        "--ros_domain_id",
        type=int,
        default=44,
        help="ROS 2 domain ID to use when initializing (default: 44)"
    )
    args = parser.parse_args()

    rclpy.init(args=None, domain_id=args.ros_domain_id)
    ctx = rclpy.get_default_context()

    config = Ros2RobotConfig()
    robot = Ros2Robot(config)
    robot.get_logger().info(f"ROS domain ID is: {ctx.get_domain_id()}")

    # Create a MultiThreadedExecutor with 4 threads
    executor = MultiThreadedExecutor(num_threads=4)

    # Add our node to it
    executor.add_node(robot)

    # Start spinning in a background thread so callbacks fire concurrently
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Once robot.disconnect() calls rclpy.shutdown(), executor.spin() will return
    spin_thread.join()


if __name__ == '__main__':
    main()
