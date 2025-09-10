import argparse
import os.path
import threading
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()

class InferenceNode(Node):
    def __init__(self, policy_path) -> None:
        super().__init__('lerobot_python')
        self.policy_path = policy_path
        self.policy = None
        self.command: int = 1 # 0: stop, 1: pause, 2: run
        self.was_paused = True
        self.shutdown = False
        self.bridge = CvBridge()
        self.state_hand_poses: Optional[Float32MultiArray] = None
        self.zed_left_color: Optional[np.ndarray] = None
        self.zed_right_color: Optional[np.ndarray] = None
        self.throttler = self.create_rate(30.0, self.get_clock())
        self.main_thread = threading.Thread(target=self.main_loop, daemon=True)

        bestEffort = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        reliable = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
        self.action_publisher = self.create_publisher(Float32MultiArray, "/lerobot/action", bestEffort)
        self.status_publisher = self.create_publisher(String, "/lerobot/status", bestEffort)
        self.left_color_subscription = self.create_subscription(Image, "/zed/color/left/image", self.left_color_callback, reliable)
        self.right_color_subscription = self.create_subscription(Image, "/zed/color/right/image", self.right_color_callback, reliable)
        self.state_hand_poses_subscription = self.create_subscription(Float32MultiArray, "/lerobot/state",
                                                                      lambda msg: setattr(self, 'state_hand_poses', msg), bestEffort)
        self.command_subscription = self.create_subscription(Int32, "/lerobot/command", self.command_callback, bestEffort)

        self.print_and_publish("Starting main loop...")
        self.main_thread.start()

    def command_callback(self, msg: Int32) -> None:
        self.command = msg.data

    def left_color_callback(self, msg: Image) -> None:
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.zed_left_color = np.transpose(cv_img, (2, 0, 1)) # OpenCV (H, W, C) -> PyTorch (C, H, W)

    def right_color_callback(self, msg: Image) -> None:
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.zed_right_color = np.transpose(cv_img, (2, 0, 1)) # OpenCV (H, W, C) -> PyTorch (C, H, W)

    def print_and_publish(self, msg: str) -> None:
        print(msg)
        self.status_publisher.publish(String(data=msg))

    def main_loop(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i} Name:", torch.cuda.get_device_name(i))
            print(f"Device {i} Properties:", torch.cuda.get_device_properties(i))

        self.print_and_publish(f"Loading policy: {self.policy_path}")
        self.policy = DiffusionPolicy.from_pretrained(str(self.policy_path))
        self.print_and_publish("Loaded policy!")
        print(self.policy)

        while not (self.command == 0 or self.shutdown):
            if (self.command != 2
             or self.state_hand_poses is None
             or self.zed_left_color is None
             or self.zed_right_color is None
            ):
                self.status_publisher.publish(String(data=f"Paused: command={self.command},"
                                                          f"state={'empty' if self.state_hand_poses is None else 'ready'},"
                                                          f"left={'empty' if self.zed_left_color is None else 'ready'},"
                                                          f"right={'empty' if self.zed_right_color is None else 'ready'}"))
                self.was_paused = True
                self.throttler.sleep()
            else:
                self.status_publisher.publish(String(data="Running!"))

                if self.was_paused:
                    self.was_paused = False
                    self.policy.reset() # Need to reset first so the actions can roll out properly

                else:
                    observation = {
                        "observation.state": torch.tensor(list(self.state_hand_poses.data), dtype=torch.float32, device=device),
                        "observation.images.cam_zed_left": torch.tensor(self.zed_left_color, dtype=torch.float32, device=device),
                        "observation.images.cam_zed_right": torch.tensor(self.zed_right_color, dtype=torch.float32, device=device),
                    }

                    with inference_mode(), autocast(device_type=device.type):
                        for name, tensor in list(observation.items()):
                            if "images" in name:
                                tensor = tensor / 255.0 # int 0-255 -> float 0.0-1.0
                            observation[name] = tensor.unsqueeze(0).to(device)
                        action = self.policy.select_action(observation).squeeze(0).detach().cpu().numpy()
                        action_hand_pose_data = action.astype(np.float32).flatten().tolist()
                        self.action_publisher.publish(Float32MultiArray(data=action_hand_pose_data))

                self.throttler.sleep()

        self.print_and_publish("Exited.")

    def stop(self) -> None:
        print("Shutting down...")
        self.shutdown = 0

if __name__ == "__main__":
    main()
