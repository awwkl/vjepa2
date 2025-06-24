import os
import glob
import pandas as pd

# Set the directory containing the training video files.
train_path_dir = './data/videos/babyview_videos'  # update this with your actual path

# Recursively gather all .mp4 files from the train_path_dir.
train_videos = glob.glob(os.path.join(train_path_dir, '**', '*.mp4'), recursive=True)
print(f"Found {len(train_videos)} training videos.")

# Write the list of video paths to a CSV file.
csv_file = './data/babyview_paths.csv'
os.makedirs(os.path.dirname(csv_file), exist_ok=True)  # Ensure the directory exists
df = pd.DataFrame(train_videos, columns=['video_path'])
df['label'] = 0  # Assign a default label value
df.to_csv(csv_file, index=False, header=False, sep=" ")

