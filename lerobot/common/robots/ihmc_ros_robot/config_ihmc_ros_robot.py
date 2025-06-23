from dataclasses import dataclass, field
from lerobot.common.robots.config import RobotConfig


from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float32MultiArray, String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
@RobotConfig.register_subclass("ros2robot")
@dataclass
class Ros2RobotConfig(RobotConfig):
    subscribers: dict[str, tuple[type, str, QoSProfile]] = field(
        default_factory=lambda: {
            '/lerobot/zed/left/color': (
                Image, '_left_color_callback',
                QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            ),
            '/lerobot/zed/right/color': (
                Image, '_right_color_callback',
                QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            ),
            '/lerobot/lerobot/state/hand_poses': (
                Float32MultiArray, '_state_hand_poses_callback',
                QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            ),
            '/lerobot/connect': (
                String, '_connect_callback',
                QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            ),
            '/lerobot/command': (
                String, '_command_callback',
                QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            ),
            '/lerobot/status': (
                String, '_status_callback',
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
    control_frequency: float   = 20.0  # Hz
    mock: bool = False