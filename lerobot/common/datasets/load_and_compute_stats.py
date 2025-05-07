from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main(dataset_path: str) -> None:
    dataset_path = Path(dataset_path)

    print(f"Dataset path: {dataset_path}")

    # Using a tolerance setting which is considerably larger than the default
    dataset = LeRobotDataset(f'root/{dataset_path.name}', dataset_path, tolerance_s=0.1)
    dataset.load_hf_dataset()
    dataset.episode_buffer = dataset.create_episode_buffer()

    # Get episode data index to iterate over episodes and frames
    episode_data_index = dataset.episode_data_index

    # Iterate over episodes
    for episode_idx in range(dataset.num_episodes):
        # Get the start and end frame indices for this episode
        start_frame_idx = episode_data_index["from"][episode_idx].item()
        end_frame_idx = episode_data_index["to"][episode_idx].item()

        # Iterate over frames in this episode
        for frame_idx in range(start_frame_idx, end_frame_idx):
            timestamp = dataset[frame_idx]["timestamp"]

            frame = {
                "action": [0, 1],
                "timestamp": 0.1,
            }

            # dataset.add_frame(frame)

    # Save episode after processing all episodes and frames
    dataset.save_episode()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
