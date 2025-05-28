#!/usr/bin/env python3
import rclpy
from pathlib import Path
import torch

from lerobot.common.robot_devices.robots.configs import Ros2RobotConfig
from lerobot.common.robot_devices.robots.Ros2Robot import Ros2Robot

def main(args=None):
    rclpy.init(args=args)

    # 1) Load the config, pointing calibration_dir at your home
    print(Path.home())
    cfg = Ros2RobotConfig(
        mock=True,
        calibration_dir=str(Path.home() / ".cache" /"calibration"),
    )
    # for demo, drive follower = leader
    cfg.follower_arms = cfg.leader_arms.copy()

    # 2) Instantiate & connect
    node = Ros2Robot(cfg)
    node.get_logger().info("Connecting…")
    node.connect()

    # 3) Send a zero‐vector action
    total_joints = sum(len(bus.motors) for bus in node.follower_arms.values())
    zero_action = torch.zeros(total_joints, dtype=torch.float32)
    node.get_logger().info(f"Sending zero action (size={total_joints})…")
    node.send_action(zero_action)

    # 4) Disconnect & shutdown
    node.get_logger().info("Disconnecting…")
    node.disconnect()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
