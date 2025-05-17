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
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video

def main():
    parser = argparse.ArgumentParser(description='Wan2.1 First-Last Frame to Video Generation')
    parser.add_argument('--first', type=str, required=True, help='First frame image path')
    parser.add_argument('--last', type=str, required=True, help='Last frame image path')
    parser.add_argument('--resolution', type=str, default='512*320', help='Output resolution in WxH format')
    parser.add_argument('--steps', type=int, default=25, help='Number of sampling steps')
    parser.add_argument('--guide_scale', type=float, default=8.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed (-1 for random)')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    
    args = parser.parse_args()
    
    # Parse resolution
    W = int(args.resolution.split('*')[0])
    H = int(args.resolution.split('*')[1])
    
    # Set seed
    seed = None if args.seed < 0 else args.seed
    
    # Initialize model
    print(f"Initializing Wan FLF2V model...")
    wan_flf2v = wan.FLF2V(
        model_id=WAN_CONFIGS['wan-flf2v-14B']['model_id'],
        device_map='cuda',
        vae_scale_factor=WAN_CONFIGS['wan-flf2v-14B']['scale_factor'],
        output_dir=os.path.dirname(args.output),
    )
    
    # Load images
    print(f"Loading first frame from: {args.first}")
    first_frame = Image.open(args.first).convert('RGB')
    
    print(f"Loading last frame from: {args.last}")
    last_frame = Image.open(args.last).convert('RGB')
    
    print(f"Generating video from first and last frames")
    video = wan_flf2v.generate(
        first_frame,
        last_frame,
        size=(W, H),
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        seed=seed,
        offload_model=True
    )
    
    # Save video
    video_path = cache_video(args.output, video)
    print(f"Video saved to: {video_path}")
    
    # Clean up
    del wan_flf2v
    
if __name__ == "__main__":
    main()
