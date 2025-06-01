"""
SUPIR Configuration Module

This module contains shared configuration and utility functions
that can be imported without triggering any GUI applications.
"""

import os
import torch
from pathlib import Path
from PIL import Image

# Device configuration
if torch.cuda.device_count() >= 2:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    SUPIR_device = 'cpu'
    LLaVA_device = 'cpu'

# Check if bf16 is supported
try:
    bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
except:
    bf16_supported = True  # Default to True for compatibility

# LLaVA model path
try:
    from SUPIR.CKPT_PTH import LLAVA_MODEL_PATH
except ImportError:
    LLAVA_MODEL_PATH = os.getenv("LLAVA_MODEL_PATH", "models/llava-v1.6-34b")

def safe_open_image(image_path):
    """Safe image opening with error handling"""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

def convert_dtype(dtype_str):
    """Convert string dtype to torch dtype"""
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        return torch.float32  # Default fallback

def get_ckpt_path(model_name: str) -> str:
    """Get the full path to a model file"""
    # Check various possible locations
    possible_paths = [
        Path("models") / "checkpoints" / model_name,
        Path("models") / model_name,
        Path("/workspace/models/checkpoints") / model_name,
        Path("/workspace/models") / model_name,
        Path("SUPIR/models") / model_name,
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return ""

def list_models():
    """List available models in the models directory"""
    models = []
    model_dirs = [
        Path("models/checkpoints"),
        Path("models"),
        Path("/workspace/models/checkpoints"),
        Path("/workspace/models"),
    ]
    
    for model_dir in model_dirs:
        if model_dir.exists():
            for file in model_dir.glob("*.safetensors"):
                models.append(file.name)
            for file in model_dir.glob("*.ckpt"):
                models.append(file.name)
    
    return list(set(models))  # Remove duplicates 