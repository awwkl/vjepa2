import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_iters_per_epoch = 300
bucket_size = 10
log_dir = 'pretrain/16.8.vitl.256px.16f/kinetics_bs512'
# log_dir = 'anneal/32.8.vitl16-256px-16f/babyview_e50'
# log_dir = 'pretrain/16.8.vitl.256px.16f/babyview_bs512'
# log_dir = 'pretrain/16.8.vitl.256px.16f/babyview_bs3072'

# log_dir = 'anneal/32.8.vitl16-256px-16f/babyview_bs3072_e60'
# log_dir = 'pretrain/16.8.vitl.256px.16f/babyview_bs3072_default'
loss_list_for_each_logfile = []
for r in range(8):
    file_path = os.path.join(log_dir, f'log_r{r}.csv')
    log_df = pd.read_csv(file_path)
    log_df = log_df[~log_df['epoch'].astype(str).str.startswith('epoch')]
    max_epoch = log_df['epoch'].astype(int).max()
    loss_list = pd.to_numeric(log_df['loss'], errors='coerce').tolist()  # Convert 'loss' column to a numeric list
    loss_list_for_each_logfile.append(loss_list)
    
# Average the losses across all logfiles

loss_list_averaged = []
min_len = min(len(l) for l in loss_list_for_each_logfile)
for i in range(min_len):
    avg_loss = sum(log[i] for log in loss_list_for_each_logfile) / len(loss_list_for_each_logfile)
    loss_list_averaged.append(avg_loss)

# group into buckets of bucket_size and average
bucket_losses = []
for i in range(0, len(loss_list_averaged), bucket_size):
    bucket = loss_list_averaged[i:i + bucket_size]
    bucket_losses.append(np.mean(bucket))

plt.plot(bucket_losses, label='Mean Loss')
plt.xticks(
    range(0, len(bucket_losses), 10 * (n_iters_per_epoch // bucket_size)),
    range(0, int(max_epoch), 10)
)
plt.ylabel('Mean Loss')
plt.title(f'Loss per Bucket -- {log_dir.split("/")[-1]}')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(log_dir, f'mean_loss_per_epoch_max_{max_epoch - 1}.png'))
plt.show()
print(f"Plot saved to {os.path.join(log_dir, f'mean_loss_per_epoch_max_{max_epoch - 1}.png')}")
