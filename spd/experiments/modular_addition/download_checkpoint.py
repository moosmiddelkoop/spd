#!/usr/bin/env python
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "gdown",
# ]
# ///

from pathlib import Path

import gdown
import torch

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Create directory structure relative to script location
checkpoint_dir = script_dir / "large_files"
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / "full_run_data.pth"

# Download the checkpoint
print(f"Downloading checkpoint to: {checkpoint_path}")
gdown.download(
    "https://drive.google.com/uc?id=12pmgxpTHLDzSNMbMCuAMXP1lE_XiCQRy",
    str(checkpoint_path),
    quiet=False
)

# Load the checkpoint
print("\nLoading checkpoint...")
data = torch.load(checkpoint_path, map_location='cpu')

# Show what's in it
print("\nKeys:", data.keys())
print("Config:", data['config'])
print("Number of checkpoints:", len(data['state_dicts']))

# Load model weights from epoch 40,000
state_dict_40k = data['state_dicts'][400]  # 400 * 100 = 40,000
print("\nLoaded weights from epoch 40,000")
print(f"Sample weight shape: {state_dict_40k['embed.W_E'].shape}")