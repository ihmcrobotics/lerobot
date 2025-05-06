from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main(dataset_path: str) -> None:
    dataset_path = Path(dataset_path)

    print(f"Dataset path: {dataset_path}")

    # Using a tolerance setting which is considerably larger than the default
    dataset = LeRobotDataset(f'root/{dataset_path.name}', dataset_path, tolerance_s=0.1)
    dataset.load_hf_dataset()
    dataset.episode_buffer = dataset.create_episode_buffer()

    for frame_index in range(dataset.num_frames):

        timestamp = dataset[frame_index]["timestamp"];

        frame = {
            "action": [0, 1],
            "timestamp": 0.1,
        }

        # dataset.add_frame(frame)


    dataset.save_episode()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])