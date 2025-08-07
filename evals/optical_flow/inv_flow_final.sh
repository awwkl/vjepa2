set -xeou pipefail 

# Run this: conda activate vjepa2-312
############

idx=3
DEVICE=7
# num=4861
num=9722
# ckpt=downloads/vitl.pt
ckpt=anneal/32.8.vitl16-256px-16f/babyview_bs3072_e60/e40.pt
model_size=vit_large

############
ZOOM_ITERS=4
STD=2
NUM_ROLLOUTS=1
fg=-1
start=$((idx * num))

SAVE_DIR=/ccn2/u/khaiaw/Code/ccwm/viz/flow_counterfactuals/full_tapvid_davis_first/std_${STD}_zoom_${ZOOM_ITERS}
datapath=/ccn2/u/ksimon12/flow/miniflow/full_tapvid_davis_first/dataset.json

# SAVE_DIR=/ccn2/u/khaiaw/Code/ccwm/viz/flow_counterfactuals/full_tapvid_kubric_first/std_${STD}_zoom_${ZOOM_ITERS}
# datapath=/ccn2/u/ksimon12/flow/miniflow/full_tapvid_kubric_first/dataset.json

cd /ccn2/u/khaiaw/Code/baselines/vjepa2

python evals/optical_flow/inv_flow_final.py \
    --out_dir=$SAVE_DIR \
    --num_rollouts $NUM_ROLLOUTS \
    --perturb_std $STD \
    --zoom_iters $ZOOM_ITERS \
    --no_blur \
    --flat_points_start_idx $start \
    --num_flat_points_to_process $num \
    --data_path $datapath \
    --model_type vjepa2 \
    --model_name $ckpt \
    --log_interval=10 \
    --frame_gap=$fg \
    --viz_interval=1000 \
    --squish \
    --device cuda:$DEVICE \
    --compile