from pathlib import Path
import cv2

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

        # Obtain the mp4 video paths for all available video keys
        for video_key in dataset.meta.video_keys:
            video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, video_key)
            print(f"Video path for {video_key}, episode {episode_idx}: {video_path}")

            # Open the video file using OpenCV
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"Error: Could not open video file {video_path}")
                else:
                    # Get video properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    print(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")

                    # Release the video capture object
                    cap.release()
            except Exception as e:
                print(f"Error processing video file {video_path}: {e}")

        # Iterate over frames in this episode
        for frame_idx in range(start_frame_idx, end_frame_idx):
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

            # FIXME Generates validate_frame error
            dataset.add_frame(frame)

    # Save episode after processing all episodes and frames
    dataset.save_episode()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
