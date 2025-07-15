# === TODOs ===
- gradient accumulation, maybe it helps stabilize training
- gradient clipping
- evaluate the pretrained released models on the same data


# === AWS ===
Security group: allow ports for ssh (22) and rsync (873)
AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
Instance type: (debugging: g4dn), (actual: p4d)
ssh -i khai.pem ubuntu@ec2______.us-west-2.compute.amazonaws.com

# === Only done once per EBS setup, because of mkfs ===
lsblk -o NAME,MAJ:MIN,SIZE,MOUNTPOINT
sudo mkfs.ext4 /dev/______

# === Mount EBS ===
EBS_MOUNT_DIR=/mnt/ebs_data
lsblk -o NAME,MAJ:MIN,SIZE,MOUNTPOINT
sudo mkdir -p $EBS_MOUNT_DIR
sudo mount /dev/______  $EBS_MOUNT_DIR
cd $EBS_MOUNT_DIR

# === Mount data drive ===
VIDEOS_DIR=/mnt/videos
sudo mkdir -p $VIDEOS_DIR
# sudo mkfs.ext4 -F -L videos /dev/_____
sudo mount /dev/_____ $VIDEOS_DIR
sudo chown "$USER":"$USER" $VIDEOS_DIR
rsync -r --info=progress2 -e "ssh -i /ccn2/u/khaiaw/Setup/aws/khai.pem" /ccn2/dataset/kinetics400/Kinetics400/k400/train/  ec2-user@ec2-54-191-18-194.us-west-2.compute.amazonaws.com:/mnt/videos/kinetics400

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
cd /ccn2/u/khaiaw/Code/baselines/vjepa2/
conda activate vjepa2-312
<!-- python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml  -->
bash train.sh

# === Cooldown ===
<!-- python -m app.main --fname configs/train/vitl16/cooldown-256px-64f.yaml -->
bash train.sh

# === To evaluate the loss of a pretrained model on the training dataset ===
python -m app.main --fname configs/train/vitl16/eval_on_train.yaml


# === Linear Probe Evaluation ===
python -m evals.main --fname configs/eval/vitl/k400.yaml  2>&1 | tee -a logs/k400_eval_$(date +%Y%m%d_%H%M%S).log
python -m evals.main --fname configs/eval/vitl/in1k.yaml --devices cuda:0 cuda:6


# === Unmount, if needed ===
sudo umount -l /mnt/ebs_data
