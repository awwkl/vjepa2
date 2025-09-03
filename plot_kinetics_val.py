import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log_dir = 'evals/vitl/k400/video_classification_frozen/babyview_pre50_cool40'
log_dir = 'evals/vitl/k400/video_classification_frozen/babyview_bs3072_pre60_cool40'
log_dir = 'evals/vitl/k400/video_classification_frozen/babyview_bs3072_pre140_cool40'
# log_dir = 'evals/vitl/k400/video_classification_frozen/babyview_bs3072_pre140_cool40_linear'
file_path = os.path.join(log_dir, f'log_r0.csv')
log_df = pd.read_csv(file_path)
log_df = log_df[~log_df['epoch'].astype(str).str.startswith('epoch')]
max_epoch = log_df['epoch'].astype(int).max()
train_acc_list = pd.to_numeric(log_df['train_acc'], errors='coerce').tolist()  # Convert 'loss' column to a numeric list
val_acc_list = pd.to_numeric(log_df['val_acc'], errors='coerce').tolist()  # Convert 'val_acc' column to a numeric list

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'epoch': log_df['epoch'],
    'train_acc': train_acc_list,
    'val_acc': val_acc_list
})
plt.figure(figsize=(10, 6))
plt.plot(plot_df['epoch'], plot_df['train_acc'], label='Train Accuracy', marker='o')
plt.plot(plot_df['epoch'], plot_df['val_acc'], label='Validation Accuracy', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'{log_dir}, max epoch {max_epoch}')
step_size = 10
if max_epoch > 200:
    step_size = 50
plt.xticks(np.arange(0, max_epoch + 1, step=step_size))
plt.legend()
plt.grid()

plt.tight_layout()
out_path = os.path.join(log_dir, f'accuracy.png')
plt.savefig(out_path)
print(f"Plot saved to {out_path}")
