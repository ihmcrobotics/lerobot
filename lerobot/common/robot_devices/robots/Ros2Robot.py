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
        self.latest_image = None   # type: Optional[np.ndarray]
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
        self.get_logger().debug(f'Published action #{self._count}: {arr.tolist()}')
        self._count += 1

    def diffusion_policy(self):
        output_directory = Path("outputs/eval/example_pusht_diffusion")
        output_directory.mkdir(parents=True, exist_ok=True)

        # Select your device
        device = "cuda"

        # Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
        pretrained_policy_path = "lerobot/diffusion_pusht"
        # OR a path to a local outputs/train folder.
        # pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

        # Initialize evaluation environment to render two observation types:
        # an image of the scene and state/position of the agent. The environment
        # also automatically stops running after 300 interactions/steps.
        env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            max_episode_steps=300,
        )

        # We can verify that the shapes of the features expected by the policy match the ones from the observations
        # produced by the environment
        print(policy.config.input_features)
        print(env.observation_space)

        # Similarly, we can check that the actions produced by the policy will match the actions expected by the
        # environment
        print(policy.config.output_features)
        print(env.action_space)

        # Reset the policy and environments to prepare for rollout
        policy.reset()
        numpy_observation, info = env.reset(seed=42)

        # Prepare to collect every rewards and all the frames of the episode,
        # from initial state to final state.
        rewards = []
        frames = []

        # Render frame of the initial state
        frames.append(env.render())

        step = 0
        done = False
        while not done:
            # Prepare observation for the policy running in Pytorch
            state = torch.from_numpy(numpy_observation["agent_pos"])
            image = torch.from_numpy(numpy_observation["pixels"])

            # Convert to float32 with image from channel first in [0,255]
            # to channel last in [0,1]
            state = state.to(torch.float32)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)

            # Send data tensors from CPU to GPU
            state = state.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)

            # Add extra (empty) batch dimension, required to forward the policy
            state = state.unsqueeze(0)
            image = image.unsqueeze(0)

            # Create the policy input dictionary
            observation = {
                "observation.state": state,
                "observation.image": image,
            }

            # Predict the next action with respect to the current observation
            with torch.inference_mode():
                action = policy.select_action(observation)

            # Prepare the action for the environment
            numpy_action = action.squeeze(0).to("cpu").numpy()

            # Step through the environment and receive a new observation
            numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
            print(f"{step=} {reward=} {terminated=}")

            # Keep track of all the rewards and frames
            rewards.append(reward)
            frames.append(env.render())

            # The rollout is considered done when the success state is reached (i.e. terminated is True),
            # or the maximum number of iterations is reached (i.e. truncated is True)
            done = terminated | truncated | done
            step += 1

        if terminated:
            print("Success!")
        else:
            print("Failure!")
    def _joint_state_callback(self, msg: JointState):
        """
        Callback invoked when a new JointState message arrives.
        Stores the latest joint positions for later retrieval.
        """
        self.latest_joints = msg
        self.get_logger().debug(f'Received joints: {msg.position}')

    def _connect_callback(self, msg: String):
        """
        Callback for the /lerobot/connect topic.
        Triggers the connection process when a message is received.
        """
        self.get_logger().info(f'Received connect: {msg.data}')
        self.connect_event.set()

    def _command_callback(self, msg: String):
        self.command = msg
        self.get_logger().debug(f'Received command: {msg.data}')
        if self.command.data == '':
            return
        elif self.command.data == 'diffusion':
            self.diffusion_policy()

    def _image_callback(self, msg: Image):
        """
        Callback invoked when a new Image message arrives.
        Converts the ROS Image message to an OpenCV image and stores it.
        """
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_image = cv_img
        self.get_logger().debug('Received new image frame')

    def _status_callback(self, msg: String):
        """
        Callback invoked when a new status message arrives.
        Sets the policy type to recieved message
        """
        self.policy_status = msg
        self.get_logger().debug(f'Received status message: {msg}')

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

    def capture_observation(self) -> Tuple[List[float], Optional[np.ndarray]]:
        # Extract joint positions if available
        if hasattr(self, 'latest_joints') and self.latest_joints is not None:
            angles: List[float] = list(self.latest_joints.position)
        else:
            angles = []
        # Extract the latest image frame if available
        image: Optional[np.ndarray] = getattr(self, 'latest_image', None)
        if image is not None and image.ndim == 3:
            # Transpose from HWC to CHW
            image = np.transpose(image, (2, 0, 1))
        return angles, image

    def disconnect(self):
        """
        Cleanly shut down the ROS2 node and stop all processing.
        Logs the shutdown and destroys the node.
        """
        self.get_logger().info('Shutting down...')
        self.is_connected = False
        self.destroy_node()
        rclpy.shutdown()