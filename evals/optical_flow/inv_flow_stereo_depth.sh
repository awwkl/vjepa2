set -xeou pipefail 

# conda activate vjepa2-312
############

idx=0
DEVICE=2
ckpt=downloads/vitl.pt
# ckpt=anneal/32.8.vitl16-256px-16f/babyview_bs3072_e60/e40.pt
# model_size=vit_large

ckpt=downloads/vitg.pt
model_size=vit_giant_xformers

############
ZOOM_ITERS=4
STD=2
NUM_ROLLOUTS=1
fg=-1
num=9999
start=$((idx * num))

SAVE_DIR=/ccn2/u/khaiaw/Code/ccwm/viz/flow_counterfactuals/flow_stereo_depth/std_${STD}_zoom_${ZOOM_ITERS}
datapath=/ccn2/u/khaiaw/Code/UniQA-3D/stereo_benchmark/metadata/flow_points.json
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
    --model_size $model_size \
    --model_name $ckpt \
    --log_interval=10 \
    --viz_interval=50 \
    --squish \
    --device cuda:$DEVICE \
    --compile