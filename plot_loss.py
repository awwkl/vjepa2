import os
import pandas as pd
import matplotlib.pyplot as plt

n_iters_per_epoch = 300
log_dir = 'pretrain/16.8.vitl.256px.16f/lr_default'
dfs = []
for r in range(8):
    file_path = os.path.join(log_dir, f'log_r{r}.csv')
    dfs.append(pd.read_csv(file_path))
log_df = pd.concat(dfs, ignore_index=True)

log_df = log_df[~log_df['epoch'].astype(str).str.startswith('epoch')]
max_epoch = log_df['epoch'].max()

log_df['overall_iter'] = log_df['epoch'] * n_iters_per_epoch + log_df['itr']
log_df['loss'] = pd.to_numeric(log_df['loss'], errors='coerce') # Ensure 'loss' is numeric before groupby to avoid aggregation errors
log_df['epoch'] = pd.to_numeric(log_df['epoch'], errors='coerce')  # Ensure 'epoch' is numeric
mean_loss_per_epoch = log_df.groupby('epoch', as_index=False)['loss'].mean()

plt.figure(figsize=(40, 5))
plt.plot(mean_loss_per_epoch['epoch'], mean_loss_per_epoch['loss'], marker='o', label='Mean Loss per Epoch')
plt.title('Mean Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Loss')
plt.xticks(mean_loss_per_epoch['epoch'])
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(log_dir, f'mean_loss_per_epoch_max_{max_epoch}.png'))
plt.show()
print(f"Plot saved to {os.path.join(log_dir, f'mean_loss_per_epoch_max_{max_epoch}.png')}")
