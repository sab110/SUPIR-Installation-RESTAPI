import os
import traceback
import torch
import numpy as np
import einops
from PIL import Image

from SUPIR.models.SUPIR_model import SUPIRModel
from SUPIR.util import HWC3, upscale_image, create_SUPIR_model
from SUPIR.utils.face_restoration_helper import FaceRestoreHelper
from SUPIR.utils.model_fetch import get_model

# Import device/dtype helpers from supir_config.py
from supir_config import SUPIR_device, bf16_supported, get_ckpt_path

# Global singletons
_model: SUPIRModel = None
_face_helper: FaceRestoreHelper = None

def _load_supir_model(model_name: str, checkpoint_type: str) -> SUPIRModel:
    """
    Lazily load (and cache) a SUPIRModel from the given checkpoint name.
    """
    global _model
    if _model is not None:
        return _model

    # Resolve checkpoint path
    ckpt_path = get_ckpt_path(model_name)
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SUPIR checkpoint not found: {model_name} -> {ckpt_path}")

    # Decide supir_sign from either model_name or checkpoint_type
    supir_sign = "Q" if "quality" in model_name.lower() or "q" in checkpoint_type.lower() else "F"

    # Choose dtype for weights
    weight_dtype = "bf16" if bf16_supported else "fp16"
    # SWITCH to DPMPP2M sampler (instead of EDM) to avoid the “got Progress” indexing bug
    sampler_target = "sgm.modules.diffusionmodules.sampling.RestoreDPMPP2MSampler"

    _model = create_SUPIR_model(
        config_path="options/SUPIR_v0.yaml",
        supir_sign=supir_sign,
        weight_dtype=weight_dtype,
        device=SUPIR_device,
        ckpt=ckpt_path,
        sampler=sampler_target
    )
    return _model

def _load_face_helper() -> FaceRestoreHelper:
    """
    Lazily load (and cache) the FaceRestoreHelper.
    """
    global _face_helper
    if _face_helper is None:
        _face_helper = FaceRestoreHelper(
            device=SUPIR_device,
            upscale_factor=1,
            face_size=512,
            use_parse=True,
            det_model='retinaface_resnet50'
        )
    return _face_helper

def run_supir(
    image_np: np.ndarray,
    *,
    checkpoint_name: str,
    checkpoint_type: str,
    upscale_size: float,
    edm_steps: int,
    s_cfg: float,
    s_stage1: float,
    s_stage2: float,
    s_churn: float,
    s_noise: float,
    color_fix_type: str,
    linear_cfg: bool,
    linear_s_stage2: bool,
    seed: int,
    num_samples: int = 1,
    p_p: str = "",            # positive prompt (usually empty for REST)
    n_p: str = "",            # negative prompt (usually empty for REST)
    cfg_scale_start: float = 0,
    control_scale_start: float = 0,
    apply_face: bool = False,
    face_prompt: str = "",
    face_resolution: int = 1024,
    max_megapixels: float = 0,
    max_resolution: int = 0
) -> np.ndarray:
    """
    Run SUPIR upscaling on a single NumPy image (H×W×3 uint8). Returns a uint8 output array.

    All named parameters match SUPIR_model.batchify_sample's signature exactly,
    but we force the DPMPP2M sampler class to avoid the “got Progress” bug.
    """
    # 1) Load (or fetch) the global model instance
    model = _load_supir_model(checkpoint_name, checkpoint_type)

    # 2) Enforce upscale constraints (mirror Gradio logic)
    h, w, _ = image_np.shape
    target_h = float(h) * float(upscale_size)
    target_w = float(w) * float(upscale_size)

    # Ensure shortest edge ≥ 1024
    if min(target_h, target_w) < 1024:
        min_scale = 1024 / min(target_h, target_w)
        target_h *= min_scale
        target_w *= min_scale

    # Enforce max megapixels
    if max_megapixels > 0:
        current_mp = (target_h * target_w) / 1e6
        if current_mp > max_megapixels:
            scale_factor = (max_megapixels * 1e6 / (target_h * target_w)) ** 0.5
            target_h *= scale_factor
            target_w *= scale_factor
            if min(target_h, target_w) < 1024:
                min_scale = 1024 / min(target_h, target_w)
                target_h *= min_scale
                target_w *= min_scale

    # Enforce max resolution
    if max_resolution > 0:
        if max(target_h, target_w) > max_resolution:
            scale_factor = max_resolution / max(target_h, target_w)
            target_h *= scale_factor
            target_w *= scale_factor
            if min(target_h, target_w) < 1024:
                min_scale = 1024 / min(target_h, target_w)
                target_h *= min_scale
                target_w *= min_scale

    # Round to multiples of 32
    unit = 32
    target_w = int(np.round(target_w / unit)) * unit
    target_h = int(np.round(target_h / unit)) * unit

    # 3) Upscale the raw numpy array to (target_h, target_w)
    hwc = HWC3(image_np)
    upscaled = upscale_image(hwc, target_w / w, unit_resolution=unit, min_size=1024)

    # Convert to SUPIR input tensor: normalize [-1,1], then to float32, shape (1,3,H,W)
    lq = (np.array(upscaled) / 255.0) * 2.0 - 1.0
    lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

    # 4) Face restoration (if requested)
    faces = []
    if apply_face:
        face_helper = _load_face_helper()
        face_helper.upscale_factor = target_h / h
        face_helper.clean_all()
        face_helper.read_image(lq.clone().cpu().squeeze(0).permute(1, 2, 0).cpu().numpy())
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()
        for cropped in face_helper.cropped_faces:
            face_arr = (np.array(cropped) / 255.0) * 2.0 - 1.0
            t = torch.tensor(face_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
            faces.append(t)

    # 5) Move model parts to correct dtype/device
    target_dtype = torch.bfloat16 if bf16_supported else torch.float16
    model.to(device=SUPIR_device)
    if hasattr(model, "model"):
        model.model = model.model.to(dtype=target_dtype, device=SUPIR_device)
    if hasattr(model, "first_stage_model"):
        model.first_stage_model = model.first_stage_model.to(dtype=target_dtype, device=SUPIR_device)
    model.ae_dtype = target_dtype

    lq = lq.to(dtype=target_dtype, device=SUPIR_device)

    # If seed < 0 → random
    if seed is None or seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    # 6) Run the sampler (DPMPP2M)
    samples = None
    try:
        samples = model.batchify_sample(
            lq,
            [face_prompt if (apply_face and face_prompt) else ""],
            num_steps=edm_steps,
            restoration_scale=s_stage1,
            s_churn=s_churn,
            s_noise=s_noise,
            cfg_scale=s_cfg,
            control_scale=s_stage2,
            seed=seed,
            num_samples=num_samples,
            p_p=p_p,
            n_p=n_p,
            color_fix_type=color_fix_type,
            use_linear_cfg=linear_cfg,
            use_linear_control_scale=linear_s_stage2,
            cfg_scale_start=cfg_scale_start,
            control_scale_start=control_scale_start,
            sampler_cls="sgm.modules.diffusionmodules.sampling.RestoreDPMPP2MSampler",
            is_cancelled=lambda: False
        )
    except Exception:
        traceback.print_exc()
        samples = None

    # 7) If SUPIR failed, fallback to bicubic
    if samples is None:
        fallback = Image.fromarray(image_np).resize(
            (int(w * upscale_size), int(h * upscale_size)),
            resample=Image.BICUBIC
        )
        return np.array(fallback)

    # 8) Convert tensor to uint8 NumPy
    clean = torch.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    x_samples = (
        einops.rearrange(clean, "b c h w -> b h w c") * 127.5 + 127.5
    ).cpu().numpy().round().clip(0, 255).astype(np.uint8)

    return x_samples[0]
