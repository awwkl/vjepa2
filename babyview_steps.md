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
# 1. rsync the mp4 videos into scratch space
rsync -rv -e "ssh -i /ccn2/u/khaiaw/Setup/aws/ondemand.pem" ______data_dir__________ root@________ip________.us-west-2.compute.amazonaws.com:/datasets/babyview_videos
# 2. create a symlink
mkdir ./data/videos
ln -s /datasets/babyview_videos ./data/videos/babyview_videos
# 3. create the dataset CSV
python data/create_train_paths_csv.py



# === Pretraining ===
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml 
