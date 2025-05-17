#!/usr/bin/env python
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import os.path as osp
import sys
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# Add model path to system path
sys.path.insert(0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import cache_video

def main():
    parser = argparse.ArgumentParser(description='Wan2.1 Image to Video Generation')
    parser.add_argument('--image', type=str, required=True, help='Image path for video generation')
    parser.add_argument('--prompt', type=str, default='', help='Text prompt for video generation')
    parser.add_argument('--resolution', type=str, default='512*320', help='Output resolution in WxH format')
    parser.add_argument('--steps', type=int, default=30, help='Number of sampling steps')
    parser.add_argument('--guide_scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--shift_scale', type=float, default=5.0, help='Shift scale')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed (-1 for random)')
    parser.add_argument('--n_prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    
    args = parser.parse_args()
    
    # Parse resolution
    W = int(args.resolution.split('*')[0])
    H = int(args.resolution.split('*')[1])
    
    # Set seed
    seed = None if args.seed < 0 else args.seed
    
    # Choose appropriate model based on resolution
    model_config = WAN_CONFIGS['wan-i2v-14B-480p'] if W * H <= MAX_AREA_CONFIGS['480P'] else WAN_CONFIGS['wan-i2v-14B-720p']
    
    # Initialize model
    print(f"Initializing Wan I2V model...")
    wan_i2v = wan.I2V(
        model_id=model_config['model_id'],
        device_map='cuda',
        vae_scale_factor=model_config['scale_factor'],
        output_dir=os.path.dirname(args.output),
    )
    
    # Load image
    print(f"Loading image from: {args.image}")
    image = Image.open(args.image).convert('RGB')
    
    print(f"Generating video from image")
    video = wan_i2v.generate(
        image,
        prompt=args.prompt,
        size=(W, H),
        shift=args.shift_scale,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        n_prompt=args.n_prompt,
        seed=seed,
        offload_model=True
    )
    
    # Save video
    video_path = cache_video(args.output, video)
    print(f"Video saved to: {video_path}")
    
    # Clean up
    del wan_i2v
    
if __name__ == "__main__":
    main()
