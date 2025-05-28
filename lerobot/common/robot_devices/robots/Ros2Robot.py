import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import torch
import json
from pathlib import Path

from lerobot.common.robot_devices.robots.configs import Ros2RobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs, MotorsBus
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs, Camera
from lerobot.common.robot_devices.motors.feetech import TorqueMode
from lerobot.common.robot_devices.robots.feetech_calibration import run_arm_manual_calibration
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError

class Ros2Robot(Node):
    def __init__(self, config: Ros2RobotConfig):
        super().__init__('ros2robot_node')
        self.config = config
        self.robot_type = config.type

        # Instantiate motors & cameras directly
        self.leader_arms = make_motors_buses_from_configs(config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(config.follower_arms)
        self.cameras      = make_cameras_from_configs(config.cameras)

        self.is_connected = False

        # A simple heartbeat publisher, just to show the node is alive
        self.pub   = self.create_publisher(String, 'robot/heartbeat', 10)
        self.timer = self.create_timer(1.0, self._heartbeat)
        self._count = 0

    def _heartbeat(self):
        msg = String()
        msg.data = f"alive {self._count}"
        self.pub.publish(msg)
        self.get_logger().info(msg.data)
        self._count += 1

    def _load_or_run_calibration(self, name: str, bus: MotorsBus, arm_type: str):
        """
        Load calibration from disk or run manual calibration if missing.
        """
        calib_dir = Path(self.config.calibration_dir)
        arm_id = get_arm_id(name, arm_type)
        path = calib_dir / f"{arm_id}.json"

        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        # else: run manual calibration
        calib = run_arm_manual_calibration(bus, self.robot_type, name, arm_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(calib, f)
        return calib

    def connect(self):
        """Bring up all motors & cameras, and run calibration."""
        # Leader arms
        for name, bus in self.leader_arms.items():
            self.get_logger().info(f"Connecting leader arm '{name}'...")
            bus.connect()
            # disable torque before calibration
            for motor_id in bus.motors:
                bus.write("Torque_Enable", TorqueMode.DISABLED.value, motor_id)
            calib = self._load_or_run_calibration(name, bus, "leader")
            bus.set_calibration(calib)

        # Follower arms
        for name, bus in self.follower_arms.items():
            self.get_logger().info(f"Connecting follower arm '{name}'...")
            bus.connect()
            # disable torque before calibration
            for motor_id in bus.motors:
                bus.write("Torque_Enable", 0, motor_id)
            # run arm-only calibration
            calib = self._load_or_run_calibration(name, bus, "follower")
            bus.set_calibration(calib)

        # Cameras (if any)
        for cam in self.cameras.values():
            cam.start()
        self.is_connected = True
        self.get_logger().info("Connected to all robot devices")

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Send a position-action to the follower arms.
        `action` is a 1D tensor whose length is the total number of motors
        across all follower arms (in the same order as config.follower_arms).
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Ros2Robot is not connected. Call connect() first.")

        action = action.flatten()
        ptr = 0
        for name, bus in self.follower_arms.items():
            n = len(bus.motors)
            slice_ = action[ptr : ptr + n].numpy().astype(float)
            bus.write("Goal_Position", slice_)
            ptr += n
            self.get_logger().info(f"Sent to '{name}': {slice_.tolist()}")
        return action

    def disconnect(self):
        """Shut down motors & cameras cleanly."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Ros2Robot is not connected.")

        # Stop follower arms (send zeros)
        for bus in self.follower_arms.values():
            zeros = [0] * len(bus.motors)
            bus.write("Goal_Position", zeros)
            bus.disconnect()

        # Disconnect leader arms
        for bus in self.leader_arms.values():
            bus.disconnect()

        # Stop cameras
        for cam in self.cameras.values():
            if isinstance(cam, Camera):
                cam.stop()

        self.is_connected = False
        self.get_logger().info("Disconnected all robot devices")