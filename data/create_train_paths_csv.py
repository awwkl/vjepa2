import os
import glob
import pandas as pd

is_eval_dataset = False
# dataset = 'kinetics400_val'
# dataset = 'kinetics'
# dataset = 'ssv2'
# dataset = 'babyview'
dataset = 'babyview_2025.2'

# Set the directory containing the training video files.
if is_eval_dataset:
    train_path_dir = f'./data/videos/eval/{dataset}_videos'  # update this with your actual path
else:
    train_path_dir = f'./data/videos/{dataset}'  # update this with your actual path

# Recursively gather all .mp4 files from the train_path_dir.
train_videos = glob.glob(os.path.join(train_path_dir, '**', '*.mp4'), recursive=True)
# train_videos = glob.glob(os.path.join(train_path_dir, '**', '*.webm'), recursive=True)
print(f"Found {len(train_videos)} training videos.")

# Write the list of video paths to a CSV file.
if is_eval_dataset:
    csv_file = f'./data/eval/{dataset}_paths.csv'
else:
    csv_file = f'./data/{dataset}_paths.csv'
os.makedirs(os.path.dirname(csv_file), exist_ok=True)  # Ensure the directory exists
df = pd.DataFrame(train_videos, columns=['video_path'])
df['label'] = 0  # Assign a default label value
df.to_csv(csv_file, index=False, header=False, sep=" ")

