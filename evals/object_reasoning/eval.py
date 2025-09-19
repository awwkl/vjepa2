"""
conda activate vjepa2-312

python evals/object_reasoning/eval.py \
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
    parser.add_argument('--input_img_dir', type=str, default='/ccn2/u/khaiaw/Code/counterfactual_benchmark/assets/ObjectReasoning', help='Directory of input images')
    parser.add_argument('--out_dir', type=str, default='/ccn2/u/khaiaw/Code/counterfactual_benchmark/model_predictions/ObjectReasoning/vjepa2', help='Directory to save output images')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4,5,6,7], help='Seeds')
    parser.add_argument('--segment_masks_dir', type=str, default='/ccn2/u/khaiaw/Code/counterfactual_benchmark/assets/ObjectReasoning/segment_masks', help='Directory of segment masks')
    parser.add_argument('--debug', action='store_true', help='Debug mode with limited data')
    return parser.parse_args()


def create_masks(unmask_indices, mask_indices):
    masks_enc = [idx + 256 for idx in unmask_indices] + list(range(256))
    masks_enc = torch.tensor(masks_enc, dtype=torch.int64).reshape(1, 1, 1, -1)
    
    masks_pred = [idx + 256 for idx in mask_indices]
    masks_pred = torch.tensor(masks_pred, dtype=torch.int64).reshape(1, 1, 1, -1)
    
    return masks_enc, masks_pred

def get_unmask_points(factual_x, factual_y):
    unmask_points = [(factual_x, factual_y)]
    
    fixed_top_points = []
    fixed_bottom_points = []
    # for every 16 range from 0 to 480,
    for i in range(32, 481, 64):
        fixed_top_points.append((i, 32))
        fixed_bottom_points.append((i, 480))
    # randomly sample N fixed_points
    num_fixed_points = fixed_point_sampler_rng.randint(3, 5)
    fixed_top_points_sampled = fixed_point_sampler_rng.sample(fixed_top_points, num_fixed_points)
    num_fixed_points = fixed_point_sampler_rng.randint(3, 5)
    fixed_bottom_points_sampled = fixed_point_sampler_rng.sample(fixed_bottom_points, num_fixed_points)
    unmask_points.extend(fixed_top_points_sampled)
    unmask_points.extend(fixed_bottom_points_sampled)
    
    return unmask_points

def convert_unmask_points_to_unmask_indices(unmask_points, patch_size, square_length_in_patches, resolution):
    num_patches_per_side = resolution // patch_size

    unmask_indices = []
    for unmask_x, unmask_y in unmask_points:
        top_left_x, top_left_y = unmask_x - patch_size * square_length_in_patches // 2, unmask_y - patch_size * square_length_in_patches // 2
        top_left_idx = (top_left_y // patch_size) * num_patches_per_side + (top_left_x // patch_size)
        patch_size_move_list = [(i, j) for i in range(square_length_in_patches) for j in range(square_length_in_patches)]
        for i, j in patch_size_move_list:
            unmask_indices.append(top_left_idx + j * num_patches_per_side + i)

    return unmask_indices

def update_pred_patch_indices(pred_patch_indices, segment_mask):
    segment_indices = set()
    segment_mask = cv2.resize(segment_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
    for i in range(segment_mask.shape[0]):
        for j in range(segment_mask.shape[1]):
            if segment_mask[i, j] == 1:
                segment_indices.add((i // 16) * 16 + (j // 16)) # based on a 16x16 patch in a 256x256 image

    updated_pred_patch_indices = [i for i in pred_patch_indices if i in segment_indices]
    return updated_pred_patch_indices

def add_factual_drawing(image, unmask_points):
    resize_transform = torchvision.transforms.Resize((512, 512))
    image = resize_transform(image)
    
    draw = ImageDraw.Draw(image)
    for i, (x, y) in enumerate(unmask_points):
        if i == 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        x = round(x / 16) * 16  # round to nearest multiple of 16
        y = round(y / 16) * 16

        draw.rectangle([x - 32, y - 32, x + 32, y + 32], outline=color, fill=None)
        draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=color) # draw a dot at x, y

    return image

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
        args.out_dir = '/ccn2/u/khaiaw/Code/counterfactual_benchmark/model_predictions/ObjectReasoning/vjepa2_debugging'
        args.seeds = [0]
    
    args.out_pred_dir = os.path.join(args.out_dir, 'pred', args.checkpoint)
    os.makedirs(args.out_pred_dir, exist_ok=True)
    
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
    annotations_csv_path = os.path.join(args.input_img_dir, 'annotations.csv')
    annotations_df = pd.read_csv(annotations_csv_path, dtype=str)
    top_keyframe_dir = os.path.join(args.input_img_dir, 'keyframes')
    
    results_df = pd.DataFrame(columns=['category', 'video_id', 'seed', 
                                       'avg_cosine_similarity_to_frame2', 'avg_cosine_similarity_to_frame3', 'closer_to_frame3',
                                       'avg_cosine_similarity_to_frame2_overall_segment', 'avg_cosine_similarity_to_frame3_overall_segment', 'closer_to_frame3_overall_segment',
                                       'avg_cosine_similarity_to_frame2_primary_segment', 'avg_cosine_similarity_to_frame3_primary_segment', 'closer_to_frame3_primary_segment'
                                       ])
    for seed in args.seeds:
        set_seed(seed)
    
        for idx, row in tqdm(annotations_df.iterrows(), total=len(annotations_df)):
            category = row['category']
            video_id = row['video_id']
            
            image_name = f'{category}_{video_id}'
            image_seed_name = f'{image_name}_seed{seed:02d}'
            
            factual_x = int(row['factual_x'])
            factual_y = int(row['factual_y'])
            
            unmask_points = get_unmask_points(factual_x, factual_y)
            unmask_points = [(x // 2, y // 2) for x, y in unmask_points] # divide unmask points by 2
            unmask_indices = convert_unmask_points_to_unmask_indices(unmask_points, 16, 2, 256)
            mask_indices = [i for i in range(256) if i not in unmask_indices]
            
            frame2_img_path = os.path.join(top_keyframe_dir, category, video_id, 'frame_02.png')
            frame3_img_path = os.path.join(top_keyframe_dir, category, video_id, 'frame_03.png')
            
            def get_video(frame2_img_path, frame3_img_path):
                """
                Load two RGB frames from disk and return a 'video' tensor shaped like Decord's
                vr.get_batch(...).asnumpy(): (T, H, W, C), dtype=uint8.
                """
                def load_rgb(path):
                    img = Image.open(path).convert("RGB")   # ensure 3 channels, RGB order
                    return np.asarray(img, dtype=np.uint8)  # (H, W, C) uint8

                f2 = load_rgb(frame2_img_path)
                f3 = load_rgb(frame3_img_path)

                # If sizes differ, resize second to match first (or raise—your call)
                if f2.shape[:2] != f3.shape[:2]:
                    Image.MAX_IMAGE_PIXELS = None
                    f3 = np.asarray(Image.fromarray(f3).resize((f2.shape[1], f2.shape[0]), Image.BILINEAR), dtype=np.uint8)

                video = np.stack([f2, f2, f3, f3], axis=0)  # repeat it 2 times because tubelet size of 2 (T=4, H, W, C)
                return video

            video = get_video(frame2_img_path, frame3_img_path) # (4, 512, 512, 3)
            video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
            
            pt_transform = build_pt_video_transform()
            clips = pt_transform(video).cuda().unsqueeze(0)
            
            clips = clips.unsqueeze(0) # [1, 1, 3, T, 256, 256]
            clips = clips.to(args.device)

            masks_enc, masks_pred = create_masks(unmask_indices, mask_indices)
            masks_enc, masks_pred = masks_enc.to(args.device), masks_pred.to(args.device)
            
            def forward_target(c):
                with torch.no_grad():
                    h = target_encoder(c)
                    h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
                    return h

            def forward_context(c):
                z = encoder(c, masks_enc)
                z = predictor(z, masks_enc, masks_pred)

                return z
            
            target = forward_target(clips)[0]  # [1, 512, 1024]
            target = target.detach().cpu()
            context = forward_context(clips)[0][0] # [1, 228, 1024]
            context = context.detach().cpu()[0]
            
            frame2_target = target[0, :target.shape[1] // 2, :] # [256, 1024]
            frame3_target = target[0, target.shape[1] // 2:, :] # [256, 1024]
            
            pred_patch_indices = masks_pred[0][0][0].tolist() # [228]
            pred_patch_indices = [pi - frame2_target.shape[0] for pi in pred_patch_indices]
            pred_patch_indices = [pi for pi in pred_patch_indices if pi >= 0] # Filter out negative indices
            
            def compute_cosine_similarities(frame2_target, frame3_target, pred_patch_indices, context):
                frame2_target_patches = frame2_target[pred_patch_indices, :]  # [N, 1024]
                frame3_target_patches = frame3_target[pred_patch_indices, :]  # [N, 1024]
                
                cosine_similarity_to_frame2 = F.cosine_similarity(frame2_target_patches, context, dim=-1)  # [N]
                avg_cosine_similarity_to_frame2 = cosine_similarity_to_frame2.mean().item()  # scalar average
                cosine_similarity_to_frame3 = F.cosine_similarity(frame3_target_patches, context, dim=-1)  # [N]
                avg_cosine_similarity_to_frame3 = cosine_similarity_to_frame3.mean().item()  # scalar average
                
                return avg_cosine_similarity_to_frame2, avg_cosine_similarity_to_frame3

            avg_cosine_similarity_to_frame2, avg_cosine_similarity_to_frame3 = compute_cosine_similarities(
                frame2_target, frame3_target, pred_patch_indices, context)
            
            
            def get_segment_indices_and_context(pred_patch_indices, segment_mask, context):
                segment_pred_patch_indices = update_pred_patch_indices(pred_patch_indices, segment_mask)
                segment_context_indices = [
                    i for i, idx in enumerate(pred_patch_indices) if idx in segment_pred_patch_indices
                ]
                segment_context = context[segment_context_indices, :]
                return segment_pred_patch_indices, segment_context

            overall_segment_mask_path = os.path.join(args.segment_masks_dir, category, video_id, 'frame3_overall_mask.npy')
            overall_segment_mask = np.load(overall_segment_mask_path)[0]
            segment_pred_patch_indices, segment_context = get_segment_indices_and_context(pred_patch_indices, overall_segment_mask, context)
            avg_cosine_similarity_to_frame2_overall_segment, avg_cosine_similarity_to_frame3_overall_segment = compute_cosine_similarities(
                frame2_target, frame3_target, segment_pred_patch_indices, segment_context)
            
            primary_segment_mask_path = os.path.join(args.segment_masks_dir, category, video_id, 'frame3_primary_mask.npy')
            primary_segment_mask = np.load(primary_segment_mask_path)[0]
            segment_pred_patch_indices, segment_context = get_segment_indices_and_context(pred_patch_indices, primary_segment_mask, context)
            avg_cosine_similarity_to_frame2_primary_segment, avg_cosine_similarity_to_frame3_primary_segment = compute_cosine_similarities(
                frame2_target, frame3_target, segment_pred_patch_indices, segment_context)

            df_row = {
                'category': category,
                'video_id': video_id,
                'seed': seed,
                'avg_cosine_similarity_to_frame2': avg_cosine_similarity_to_frame2,
                'avg_cosine_similarity_to_frame3': avg_cosine_similarity_to_frame3,
                'closer_to_frame3': bool(avg_cosine_similarity_to_frame2 <= avg_cosine_similarity_to_frame3),
                'avg_cosine_similarity_to_frame2_overall_segment': avg_cosine_similarity_to_frame2_overall_segment,
                'avg_cosine_similarity_to_frame3_overall_segment': avg_cosine_similarity_to_frame3_overall_segment,
                'closer_to_frame3_overall_segment': bool(avg_cosine_similarity_to_frame2_overall_segment <= avg_cosine_similarity_to_frame3_overall_segment),
                'avg_cosine_similarity_to_frame2_primary_segment': avg_cosine_similarity_to_frame2_primary_segment,
                'avg_cosine_similarity_to_frame3_primary_segment': avg_cosine_similarity_to_frame3_primary_segment,
                'closer_to_frame3_primary_segment': bool(avg_cosine_similarity_to_frame2_primary_segment <= avg_cosine_similarity_to_frame3_primary_segment),
            }
            results_df = pd.concat([results_df, pd.DataFrame([df_row])], ignore_index=True)
            
    # for each category, print the average for closer_to_frame3, closer_to_frame3_overall_segment, closer_to_frame3_primary_segment
    avg_results = results_df.groupby('category').agg({
        'closer_to_frame3': 'mean',
        'closer_to_frame3_overall_segment': 'mean',
        'closer_to_frame3_primary_segment': 'mean'
    }).reset_index()
    avg_results[['closer_to_frame3', 'closer_to_frame3_overall_segment', 'closer_to_frame3_primary_segment']] *= 100
    avg_results.columns = ['category', 'avg_closer_to_frame3', 'avg_closer_to_frame3_overall_segment', 'avg_closer_to_frame3_primary_segment']
    results_df = pd.merge(results_df, avg_results, on='category', how='left')
    results_df.to_csv(os.path.join(args.out_pred_dir, f'results.csv'), index=False)
    pprint.pprint(avg_results)

    # ===== Extra shapes for comments purposes =====
    # clips: [ [64, 3, 2, 256, 256] ]
    # h: [ [64, 256 ????, 1024] ]
    
    # masks_enc[0][0]: [64, 130]
    # masks_enc[0][1]: [64, 32]
    # masks_pred[0][0]: [64, 228]
    # masks_pred[0][1]: [64, 352]
    # z[0][0]: [64, 228, 1024]
    # z[0][1]: [64, 352, 1024]

    # masks_enc[0][0][0] and masks_pred[0][0][0] -- check if they share any numbers
    # shared = set(masks_enc[0][0][0].tolist()).intersection(set(masks_pred[0][0][0].tolist()))
    # check length of union
    # union = set(masks_enc[0][0][0].tolist()).union(set(masks_pred[0][0][0].tolist()))