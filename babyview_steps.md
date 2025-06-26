# === AWS ===
Security group: allow ports for ssh (22) and rsync (873)
AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
Instance type: (debugging: g4dn), (actual: p4d)
ssh -i khai.pem ubuntu@ec2______.us-west-2.compute.amazonaws.com

# === Only done once per EBS setup, because of mkfs ===
lsblk -o NAME,MAJ:MIN,SIZE,MOUNTPOINT
sudo mkfs.ext4 /dev/______

# === Mount EBS ===
EBS_MOUNT_DIR=$/mnt/ebs_data
lsblk -o NAME,MAJ:MIN,SIZE,MOUNTPOINT
sudo mkdir -p $EBS_MOUNT_DIR
sudo mount /dev/______ 
cd $EBS_MOUNT_DIR

# === Create a scratch drive and dir ===
SCRATCH_DRIVE=/dev/nvme1n1
SCRATCH_DIR=/data/videos
sudo mkdir -p $SCRATCH_DIR
sudo mount $SCRATCH_DRIVE $SCRATCH_DIR
sudo chown "$USER":"$USER" $SCRATCH_DIR
export TMPDIR=$SCRATCH_DIR                      # Change build directory for pip / conda
export PIP_CACHE_DIR=$SCRATCH_DIR

# === Enter the directory ===
git clone https://github.com/awwkl/vjepa2.git
cd vjepa2



# === Install conda to EBS mount, only done once ===
mkdir -p $EBS_MOUNT_DIR/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $EBS_MOUNT_DIR/miniconda3/miniconda.sh
bash $EBS_MOUNT_DIR/miniconda3/miniconda.sh -b -u -p $EBS_MOUNT_DIR/miniconda3
rm $EBS_MOUNT_DIR/miniconda3/miniconda.sh
source $EBS_MOUNT_DIR/miniconda3/bin/activate
conda init --all

conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
pip install .

# Shell hook to activate conda, (maybe needed if using this EBS block from elsewhere)
source $EBS_MOUNT_DIR/miniconda3/etc/profile.d/conda.sh


# === Data ===
# 1. rsync the mp4 videos into scratch space
rsync -rv -e "ssh -i /ccn2/u/khaiaw/Setup/aws/khai.pem" /ccn2/dataset/kinetics400/Kinetics400/k400/train/ ec2-user@ec2-35-90-7-251.us-west-2.compute.amazonaws.com:/data/videos
# 2. create a symlink
mkdir ./data/videos
ln -s $SCRATCH_DIR ./data/videos/babyview_videos
# 3. create the dataset CSV
python data/create_train_paths_csv.py



# === Pretraining ===
tmux new -s train
conda activate vjepa2-312
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml 


# === Unmount, if needed ===
sudo umount -l /mnt/ebs_data
