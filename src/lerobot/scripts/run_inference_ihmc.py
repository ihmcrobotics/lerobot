import argparse
import os.path
import threading
import time
from typing import Optional

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String, Int32
from torch import autocast, inference_mode

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch IHMC ROS 2 inference."
    )
    parser.add_argument(
        "--policy",
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

    node = InferenceNode(args.policy)
    rclpy.spin(node)

class InferenceNode(Node):
    def __init__(self, policy_path) -> None:
        super().__init__('lerobot_python')
        self.command: int = 1 # 0: stop, 1: pause, 2: run
        self.bridge = CvBridge()
        self.state_hand_poses: Optional[Float32MultiArray] = None
        self.zed_left_color: Optional[np.ndarray] = None
        self.zed_right_color: Optional[np.ndarray] = None
        self.rate = self.create_rate(50.0, self.get_clock()) # tool to sleep at 50 Hz

        print(f"Loading policy: {policy_path}")
        self.policy = DiffusionPolicy.from_pretrained(
            str(policy_path),
            local_files_only=False,
        )

        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT) # Match IHMC default
        self.action_publisher = self.create_publisher(Float32MultiArray, "/lerobot/action", qos)
        self.status_publisher = self.create_publisher(String, "/lerobot/status", qos)
        self.left_color_subscription = self.create_subscription(Image, "/zed/color/left/image", self.left_color_callback, qos)
        self.right_color_subscription = self.create_subscription(Image, "/zed/color/right/image", self.right_color_callback, qos)
        self.state_hand_poses_subscription = self.create_subscription(Float32MultiArray, "/lerobot/state",
                                                                      lambda msg: setattr(self, 'state_hand_poses', msg), qos)
        self.command_subscription = self.create_subscription(Int32, "/lerobot/command", lambda msg: setattr(self, 'command', msg), qos)

        threading.Thread(target=self.main_loop, daemon=True).start()

    def left_color_callback(self, msg: Image) -> None:
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.zed_left_color = np.transpose(cv_img, (2, 0, 1))

    def right_color_callback(self, msg: Image) -> None:
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.zed_right_color = np.transpose(cv_img, (2, 0, 1))

    def main_loop(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.status_publisher.publish(String(data=f"Initialized {device}"))

        self.policy.to(device)
        self.policy.reset()
        self.policy.eval()

        while not self.command == 0:
            if (self.command != 2
                or self.state_hand_poses is None
                or self.zed_left_color is None
                or self.zed_right_color is None
            ):
                self.rate.sleep()
            else:
                self.status_publisher.publish(String(data="Running"))

                observation = {
                    "observation.state": torch.tensor(list(self.state_hand_poses.data), dtype=torch.float32, device=device),
                    "observation.images.cam_zed_left": torch.tensor(self.zed_left_color, dtype=torch.float32, device=device),
                    "observation.images.cam_zed_right": torch.tensor(self.zed_right_color, dtype=torch.float32, device=device),
                }

                with inference_mode(), autocast(device_type=device.type):
                    for name, tensor in list(observation.items()):
                        if "images" in name:
                            tensor = tensor / 255.0 # normalize
                            if tensor.ndim == 3 and tensor.shape[2] in (1, 3):
                                tensor = tensor.permute(2, 0, 1).contiguous()
                        observation[name] = tensor.unsqueeze(0).to(device)
                    action = self.policy.select_action(observation).squeeze(0).detach().cpu().numpy()
                    action_hand_pose_data = action.astype(np.float32).flatten().tolist()
                    self.action_publisher.publish(Float32MultiArray(data=action_hand_pose_data))

                self.rate.sleep()

        self.status_publisher.publish(String(data="Exited"))
        time.sleep(0.5)

        self.get_logger().info("Shutting down ROS 2 node...")
        self.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
