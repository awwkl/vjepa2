# === AWS ===
Allow ports for ssh (22) and rsync (873)

# === Setup EBS for training ===
lsblk -o NAME,MAJ:MIN,SIZE,MOUNTPOINT
sudo mkfs.ext4 /dev/nvme1n1

# === Mount EBS 
lsblk -o NAME,MAJ:MIN,SIZE,MOUNTPOINT
sudo mkdir -p /mnt/ebs_data 
sudo mount /dev/nvme1n1 /mnt/ebs_data
cd /mnt/ebs_data


# === Enter the directory ===
git clone https://github.com/awwkl/vjepa2.git
cd /mnt/ebs_data/vjepa2


# === Install ===
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
pip install .  # or `pip install -e .` for development mode

# === Data ===
# 1. rsync the mp4 videos into EFS
# 2. rsync the data csv into ./data/babyview_paths.csv
# 3. create a symlink
mkdir ./data/videos
ln -s ________ ./data/videos/babyview_videos



# === Pretraining ===
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml 
