import time
import rclpy
from rclpy.node import Node
import torch
import numpy as np
import threading

from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
from typing import Optional, Tuple, List
from pathlib import Path
from lerobot.common.robot_devices.robots.configs import Ros2RobotConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

class Ros2Robot(Node):
    def __init__(self, config: Ros2RobotConfig):
        """
        Initialize the ROS2 robot node.
        Sets up publishers for actions, subscribers for joint states and images,
        and state variables to track the latest observations.
        """
        super().__init__('ros2robot')
        self.config = config

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


        # Initialize state variables
        self.command = ""
        self.connect_event = threading.Event()
        self.policy_status = ""
        self.state_hand_poses = None
        self.right_color = None
        self.left_color = None
        self._count = 0
        self.is_connected = False
        self.diffusion_Start = False

    def connect(self):
        """
        Mark the robot as connected and enable sending actions.
        Logs connection status to the ROS2 console.
        """
        self.get_logger().info('Waiting for /lerobot/connect to connect...')
        # self.connect_event.wait()  # Blocks here until _connect_callback sets the event
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
        self.lerobot_lerobot_action_hand_poses_pub.publish(msg)

        self._count += 1

    def status_thread(self):
        """
        Thread for sending status through ROS.
        """
        while self.is_connected:
            status = String()
            status.data = "True"
            self.lerobot_status_pub.publish(status)
            rclpy.spin_once(self, timeout_sec=0.05)
        status = String()
        status.data = "False"
        self.lerobot_status_pub.publish(status)
        rclpy.spin_once(self, timeout_sec=0.05)
    def run_diffusion_policy(self, max_steps=100, policy_path=Path("outputs/train/pretrained_model")):
        """
        Runs the diffusion policy on the real robot through ROS2 topics,

        Args:
            max_steps (int): Number of steps to run.
            policy_path (str): HF repo ID or local path for the pretrained policy.
        """

        self.diffusion_Start = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = DiffusionPolicy.from_pretrained(policy_path)
        policy.to(device)
        policy.reset()
        policy.eval()

        self.get_logger().info(f"Running diffusion policy for up to {max_steps} steps...")

        step = 0
        while step < max_steps and self.is_connected:
            if not (self.state_hand_poses and self.left_color is not None and self.right_color is not None):
                continue

            state_hand_poses, left_color, right_color = self.capture_observation()
            state = torch.tensor(state_hand_poses, dtype=torch.float32, device=device).unsqueeze(0)
            img_tensor_left = torch.tensor(left_color, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            img_tensor_right = torch.tensor(right_color, dtype=torch.float32, device=device).unsqueeze(0) / 255.0

            obs = {
                "observation.state": state,
                "observation.images.cam_zed_left": img_tensor_left,
                "observation.images.cam_zed_right": img_tensor_right,
            }

            with torch.no_grad():
                action = policy.select_action(obs)
            action = action.squeeze(0)

            self.send_action(action)

            self.get_logger().info(f'Published action #{step}')
            step += 1

        self.get_logger().info("Diffusion policy run complete or robot disconnected.")
        time.sleep(1)
        self.disconnect()

    def _state_hand_poses_callback(self, msg: Float32MultiArray):
        """
        Callback invoked when a new JointState message arrives.
        Stores the latest joint positions for later retrieval.
        """
        self.state_hand_poses = msg

    def _connect_callback(self, msg: String):
        """
        Callback for the /lerobot/connect topic.
        Triggers the connection process when a message is received.
        """
        self.connect_event.set()

    def _command_callback(self, msg: String):
        '''
        Callback for the /lerobot/command topic.
        Switches from no policy to diffusion policy.
        '''
        self.command = msg
        if self.diffusion_Start:
            return
        if self.command.data == '':
            return
        elif self.command.data == 'diffusion':
            # Launch diffusion in separate thread
            self.get_logger().info(f'Received diffusion command: {msg.data}')
            threading.Thread(target=self.run_diffusion_policy, daemon=True).start()
        else:
            self.get_logger().info(f'Command callback is: {msg.data}')
            self.connect_event.set()

    def _left_color_callback(self, msg: Image):
        """
        Callback invoked when a new Image message arrives.
        Converts the ROS Image message to an OpenCV image and stores it.
        """
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("Got left color")
        self.left_color = cv_img
    def _right_color_callback(self, msg: Image):
        """
        Callback invoked when a new Image message arrives.
        Converts the ROS Image message to an OpenCV image and stores it.
        """
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("Got right color")
        self.right_color = cv_img

    def _status_callback(self, msg: String):
        """
        Callback invoked when a new status message arrives.
        Sets the policy type to recieved message
        """
        self.policy_status = msg.data

    def capture_observation(self) -> Tuple[List[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if hasattr(self, 'state_hand_poses') and self.state_hand_poses is not None:
            angles: List[float] = list(self.state_hand_poses.data)
        else:
            angles = None

        image_left: Optional[np.ndarray] = self.left_color
        if image_left is not None and image_left.ndim == 3:
            image_left = np.transpose(image_left, (2, 0, 1))

        image_right: Optional[np.ndarray] = self.right_color
        if image_right is not None and image_right.ndim == 3:
            image_right = np.transpose(image_right, (2, 0, 1))

        return angles, image_left, image_right

    def disconnect(self):
        """
        Cleanly shut down the ROS2 node and stop all processing.
        Logs the shutdown and destroys the node.
        """
        self.get_logger().info('Shutting down...')
        self.is_connected = False
        status = String()
        status.data = "False"
        self.lerobot_status_pub.publish(status)
        time.sleep(2)
        self.destroy_node()
        rclpy.shutdown()

def main():
    import os
    os.environ['ROS_DOMAIN_ID'] = '185'
    rclpy.init()
    config = Ros2RobotConfig()
    robot = Ros2Robot(config)
    spin_thread = threading.Thread(target=rclpy.spin, args=(robot,))
    spin_thread.start()
    status_thread = threading.Thread(target=robot.status_thread, args=())
    robot.connect()
    status_thread.start()



if __name__ == '__main__':
    main()
