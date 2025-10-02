"""
conda activate vjepa2-312

CUDA_VISIBLE_DEVICES=5 \
python evals/extract_activations/extraction_activations.py \
    --image_dir /ccn2a/dataset/babyview/2025.2/extracted_frames_1fps/S00220001_2024-02-05_2_recQNiTr64Y9FpiPW/ \
    --forward_mode two_image \
    --output_activations_dir viz/extract_activations/activations/ \
    --checkpoint downloads/vitl.pt \
        
        
    --checkpoint anneal/32.8.vitl16-256px-16f/babyview_bs3072_e140/e40.pt \
    --checkpoint downloads/vitg.pt \
    --model_name vit_giant_xformers \
    --checkpoint downloads/vith.pt \
    --model_name vit_huge \

Referenced code from:
- evals/video_classification_frozen/eval.py (for the model initialization and loading functions)
- notebooks/vjepa2_demo.py (for the video processing transforms)

"""

import glob
import sys
import os
import random
import argparse
import yaml
import copy
from tqdm import tqdm
import pprint

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import cv2
import torch
import torchvision
import torch.nn.functional as F


# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.vjepa.utils import init_opt, init_video_model, load_checkpoint
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms


fixed_point_sampler_rng = None # use a private generator to sample fixed points, unaffected by model calls
def set_seed(seed):
    global fixed_point_sampler_rng
    fixed_point_sampler_rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--model_name', default='vit_large', help='Model size to use', choices=['vit_large', 'vit_huge', 'vit_giant_xformers'])
    parser.add_argument("--fname", type=str, help="name of config file to load", default="./evals/object_reasoning/config.yaml")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--output_activations_dir', type=str, default='viz/extract_activations/activations', help='Directory to save extracted activations')
    parser.add_argument('--forward_mode', type=str, default='one_image',choices=['one_image', 'two_image', 'four_image', 'eight_image'], help='Which part of the model to use for forward pass')
    parser.add_argument('--n_samples_to_eval', type=int, default=10, help='Number of samples to evaluate')
    parser.add_argument('--out_dir', type=str, default='viz/extract_activations', help='Directory to save output images')
    parser.add_argument('--debug', action='store_true', help='Debug mode with limited data')
    return parser.parse_args()

def build_pt_video_transform():
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(256, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(256, 256)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform

if __name__ == "__main__":
    args = get_args()
    
    if args.debug:
        args.n_samples_to_eval = 10
    
    config_params = None
    with open(args.fname, "r") as y_file:
        config_params = yaml.load(y_file, Loader=yaml.FullLoader)
    print(config_params)
    print(args)
    
    cfgs_model = config_params.get("model")
    cfgs_model['model_name'] = args.model_name
    cfgs_mask = config_params.get("mask")
    cfgs_data = config_params.get("data")
    cfgs_meta = config_params.get("meta")
    cfgs_opt = config_params.get("optimization")
    
    # === Initialize model (without weights) ===
    encoder, predictor = init_video_model(
        uniform_power=cfgs_model.get("uniform_power", False),
        use_mask_tokens=cfgs_model.get("use_mask_tokens", False),
        num_mask_tokens=int(len(cfgs_mask) * len(cfgs_data.get("dataset_fpcs"))),
        zero_init_mask_tokens=cfgs_model.get("zero_init_mask_tokens", True),
        device=args.device,
        patch_size=cfgs_data.get("patch_size"),
        max_num_frames=max(cfgs_data.get("dataset_fpcs")),
        tubelet_size= cfgs_data.get("tubelet_size"),
        model_name=cfgs_model.get("model_name"),
        crop_size=256,
        pred_depth=cfgs_model.get("pred_depth"),
        pred_num_heads=cfgs_model.get("pred_num_heads", None),
        pred_embed_dim=cfgs_model.get("pred_embed_dim"),
        use_sdpa=cfgs_meta.get("use_sdpa", False),
        use_silu=cfgs_model.get("use_silu", False),
        use_pred_silu=cfgs_model.get("use_pred_silu", False),
        wide_silu=cfgs_model.get("wide_silu", True),
        use_rope=cfgs_model.get("use_rope", False),
        use_activation_checkpointing=cfgs_model.get("use_activation_checkpointing", False),
    )
    target_encoder = copy.deepcopy(encoder)

    # === Load model weights from checkpoint ===
    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    def load_state(model, key):
        state = checkpoint[key]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    load_state(encoder, "encoder")
    load_state(predictor, "predictor")
    load_state(target_encoder, "target_encoder")
    
    # === Load images into correct format ===
    image_files = glob.glob(os.path.join(args.image_dir, '**/*.jpg'), recursive=True)
    random.shuffle(image_files)

    for i, image_file in enumerate(tqdm(image_files, total=args.n_samples_to_eval)):
        if i >= args.n_samples_to_eval:
            break
        
        image_name = os.path.basename(image_file).split('.')[0]
        image = Image.open(image_file).convert("RGB")
        
        def get_video(frame2_img_path, forward_mode):
            """
            Load two RGB frames from disk and return a 'video' tensor shaped like Decord's
            vr.get_batch(...).asnumpy(): (T, H, W, C), dtype=uint8.
            """
            def load_rgb(path):
                img = Image.open(path).convert("RGB")   # ensure 3 channels, RGB order
                return np.asarray(img, dtype=np.uint8)  # (H, W, C) uint8

            f2 = load_rgb(frame2_img_path)
            
            if forward_mode == 'one_image':
                num_repeat_times = 1 * 2
            elif forward_mode == 'two_image':
                num_repeat_times = 2 * 2
            elif forward_mode == 'four_image':
                num_repeat_times = 4 * 2
            elif forward_mode == 'eight_image':
                num_repeat_times = 8 * 2

            video = np.stack([f2] * num_repeat_times, axis=0)  # repeat it 2 times because tubelet size of 2 (T=4, H, W, C)
            return video

        video = get_video(image_file, forward_mode=args.forward_mode) # (2, 512, 512, 3)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        
        pt_transform = build_pt_video_transform()
        clips = pt_transform(video).cuda().unsqueeze(0)
        
        clips = clips.unsqueeze(0) # [1, 1, 3, T, 256, 256]
        clips = clips.to(args.device)

        def forward_target(c):
            with torch.no_grad():
                h = target_encoder(c)
                h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
                return h
        
        target = forward_target(clips)[0]  # [1, 512, 1024]
        target = target.detach().cpu()[0] # [512, 1024]
        
        activations = target
        print(activations.shape)
        
        # === Save ===
        act_save_dir = os.path.join(args.output_activations_dir, f"vjepa2_{args.forward_mode}--{args.checkpoint.replace('/', '_')}")
        act_save_path = os.path.join(act_save_dir, f'{image_name}.npy')
        os.makedirs(act_save_dir, exist_ok=True)
        np.save(act_save_path, activations.numpy())
        