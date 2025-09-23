import os
import glob
import pandas as pd
from tqdm import tqdm

val_or_train = 'train'  # Change to 'train' if needed

videos_dir = f'./data/videos/kinetics400_{val_or_train}'
# if val_or_train == 'train':
#     videos_dir = f'./data/videos/kinetics_videos'
# else:
#     videos_dir = f'./data/videos/eval/kinetics400_{val_or_train}_videos'

annotations_file = f'./data/kinetics/annotations/{val_or_train}.csv'
class_name_to_number_file = './data/kinetics/annotations/kinetics_400_labels.csv'
output_csv_path = f'./data/kinetics/kinetics400_{val_or_train}_paths.csv'

# Get video paths and extract YouTube IDs
videos_paths = glob.glob(os.path.join(videos_dir, '**', '*.mp4'), recursive=True)
print(f"Found {len(videos_paths)} videos")

# Create DataFrame with video paths and extract YouTube IDs
output_df = pd.DataFrame(videos_paths, columns=['video_path'])
output_df['youtube_id'] = output_df['video_path'].apply(lambda x: os.path.basename(x)[:11])

# Load annotations and class mappings
annotations_df = pd.read_csv(annotations_file)
class_name_to_number_df = pd.read_csv(class_name_to_number_file)
class_name_to_number = dict(zip(class_name_to_number_df['name'], class_name_to_number_df['id']))

print(f"Loaded {len(annotations_df)} annotations")
print(f"Loaded {len(class_name_to_number)} class mappings")

# Vectorized merge operation - this is the key optimization
output_df = output_df.merge(
    annotations_df[['youtube_id', 'label']], 
    on='youtube_id', 
    how='left'
)

# Map class names to numbers using vectorized operation
output_df['label'] = output_df['label'].map(class_name_to_number).fillna(-1).astype(int)

# Drop the helper column
output_df = output_df.drop('youtube_id', axis=1)

# Report missing mappings
missing_count = (output_df['label'] == -1).sum()
if missing_count > 0:
    print(f"Warning: {missing_count} videos could not be mapped to labels")

# Save results
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
output_df.to_csv(output_csv_path, index=False, header=False, sep=" ")

print(f"Saved results to {output_csv_path}")
print(f"Total videos processed: {len(output_df)}")
print(f"Successfully labeled: {len(output_df) - missing_count}")