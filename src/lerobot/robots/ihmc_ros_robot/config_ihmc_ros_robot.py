from dataclasses import dataclass, field
from lerobot.robots.config import RobotConfig

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


@RobotConfig.register_subclass("ihmc_ros_robot")
@dataclass
class Ros2RobotConfig(RobotConfig):
    subscribers: dict[str, tuple[type, str, QoSProfile]] = field(
        default_factory=lambda: {
            '/zed/color/left/image': (
                Image, 'left_color_callback',
                QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            ),
            '/zed/color/right/image': (
                Image, 'right_color_callback',
                QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            ),
            '/lerobot/lerobot/state/hand_poses': (
                Float32MultiArray, 'state_hand_poses_callback',
                QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            ),
            '/lerobot/command': (
                String, 'command_callback',
                QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            ),
            '/lerobot/status': (
                String, 'status_callback',
                QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            ),
        }
    )
    publishers: dict[str, tuple[type, int]] = field(
        default_factory=lambda: {
            '/lerobot/lerobot/action/hand_poses': (Float32MultiArray, 10),
            '/lerobot/status': (String, 10),
        }
    )
    control_frequency: float = 20.0  # Hz
    mock: bool = False
