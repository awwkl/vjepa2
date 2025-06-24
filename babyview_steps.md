# === Enter the directory ===
cd /ccn2/u/khaiaw/Code/baselines/vjepa2/

# === Install ===
conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
pip install .  # or `pip install -e .` for development mode

# === Data ===
# 1. rsync the mp4 videos into EFS
# 2. rsync the data csv into ./data/babyview_paths.csv
# 3. create a symlink
ln -s /ccn2/dataset/babyview/unzip_2025_10s_videos_256p/ ./data/babyview_videos



# === Pretraining ===
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml 

