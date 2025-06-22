from pathlib import Path

from lerobot.common.datasets.compute_stats import compute_episode_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_episode_stats

def main(dataset_path: str) -> None:
    dataset_path = Path(dataset_path)

    print(f"Dataset path: {dataset_path}")

    # Using a tolerance setting which is considerably larger than the default
    dataset = LeRobotDataset(f'root/{dataset_path.name}', dataset_path, tolerance_s=0.1)
    dataset.load_hf_dataset()

    # Get episode data index to iterate over episodes and frames
    episode_data_index = dataset.episode_data_index

    # Iterate over episodes
    num_episodes = dataset.num_episodes
    for episode_idx in range(num_episodes):
        print(f"Proccessing episode: {episode_idx} / {num_episodes - 1}...")

        # Create a new episode buffer with the correct episode index
        dataset.episode_buffer = dataset.create_episode_buffer(episode_idx)

        # Get the start and end frame indices for this episode
        start_frame_idx = episode_data_index["from"][episode_idx].item()
        end_frame_idx = episode_data_index["to"][episode_idx].item()

        # Iterate over frames in this episode
        for frame_idx in range(start_frame_idx, end_frame_idx):
            print(f"\rFrame: {start_frame_idx} -> {frame_idx} / {end_frame_idx}", end="", flush=True)

            # Get the frame data from the dataset
            frame_data = dataset[frame_idx]
            timestamp = frame_data["timestamp"]

            # Create a complete frame dictionary with all necessary data
            frame = {}

            # Add all keys from frame_data except those that are automatically handled
            for key in frame_data:
                # Skip keys that are automatically handled by add_frame
                if key not in ["index", "episode_index", "frame_index", "task_index"]:
                    frame[key] = frame_data[key]

            # Make sure to use the actual timestamp
            frame["timestamp"] = timestamp

            # Add task if available
            if "task" in frame_data:
                frame["task"] = frame_data["task"]

            # Add frame to the current episode buffer
            dataset.add_frame(frame)

        print("\nDone adding frames.")

        # Compute episode stats
        ep_stats = compute_episode_stats(dataset.episode_buffer, dataset.features)

        # Save episode stats
        # write_episode_stats(episode_idx, ep_stats, dataset_path)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
