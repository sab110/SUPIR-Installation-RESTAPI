"""
SUPIR REST API Server

Provides REST endpoints for image upscaling using SUPIR:
- POST /job: Submit a new upscaling job
- GET /job/{job_id}: Get job status
- GET /job/{job_id}/result: Download the upscaled image

Compatible with RunPod and designed for network storage integration.
"""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from fastapi import FastAPI, File, Form, HTTPException, Depends, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

# Import SUPIR components
from supir_utils import run_supir
from SUPIR.models.SUPIR_model import SUPIRModel
from SUPIR.util import HWC3, upscale_image, convert_dtype, create_SUPIR_model
from SUPIR.utils.status_container import StatusContainer, MediaData
from SUPIR.utils.face_restoration_helper import FaceRestoreHelper
from SUPIR.utils.model_fetch import get_model
# from supir_config import SUPIR_device, LLaVA_device, bf16_supported, LLAVA_MODEL_PATH, get_ckpt_path, list_models
from llava.llava_agent import LLavaAgent
import ui_helpers
from ui_helpers import printt

# Import configuration without triggering Gradio
try:
    from supir_config import (
        SUPIR_device, LLaVA_device, bf16_supported, LLAVA_MODEL_PATH,
        safe_open_image, convert_dtype, get_ckpt_path, list_models
    )
except ImportError:
    # Define fallbacks if supir_config is not available
    print("Warning: supir_config not found, using fallback configuration")
    
    # Device configuration fallback
    if torch.cuda.device_count() >= 2:
        SUPIR_device = 'cuda:0'
        LLaVA_device = 'cuda:1'
    elif torch.cuda.device_count() == 1:
        SUPIR_device = 'cuda:0'
        LLaVA_device = 'cuda:0'
    else:
        SUPIR_device = 'cpu'
        LLaVA_device = 'cpu'
    
    bf16_supported = True
    LLAVA_MODEL_PATH = os.getenv("LLAVA_MODEL_PATH", "models/llava-v1.6-34b")
    
    def safe_open_image(image_path):
        """Fallback implementation of safe_open_image"""
        try:
            from PIL import Image
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
    
    def get_ckpt_path(model_name):
        return get_model_path(model_name)
    
    def list_models():
        return []

# Global variables
app = FastAPI(
    title="SUPIR API",
    description="REST API for SUPIR image upscaling",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
security = HTTPBearer()
API_TOKEN = os.getenv("API_TOKEN", "your-secret-token")

# Global state
model: Optional[SUPIRModel] = None
llava_agent: Optional[LLavaAgent] = None
face_helper: Optional[FaceRestoreHelper] = None
jobs: Dict[str, Dict] = {}

# Paths
WORKSPACE_PATH = Path("/workspace")
INPUT_FOLDER = WORKSPACE_PATH / "adjusted"
OUTPUT_FOLDER = WORKSPACE_PATH / "adjustedupscaled"
MODELS_PATH = WORKSPACE_PATH / "models"
TEMP_UPLOAD_PATH = WORKSPACE_PATH / "temp"

# Create directories
for path in [INPUT_FOLDER, OUTPUT_FOLDER, MODELS_PATH, TEMP_UPLOAD_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Pydantic models
class JobSettings(BaseModel):
    upscale_size: int = Field(default=2, ge=1, le=8, description="Upscale factor (1-8)")
    apply_llava: bool = Field(default=True, description="Apply LLaVa for captioning")
    apply_supir: bool = Field(default=True, description="Apply SUPIR upscaling")
    prompt_style: str = Field(default="Photorealistic", description="Style of the prompt")
    model: str = Field(default="RealVisXL_V5.0_fp16.safetensors", description="Model to use")
    checkpoint_type: str = Field(default="Standard SDXL", description="Checkpoint type")
    prompt: str = Field(default="", description="Custom prompt")
    save_captions: bool = Field(default=True, description="Save captions to text files")
    text_guidance_scale: int = Field(default=1024, ge=512, le=2048, description="Text guidance scale")
    background_restoration: bool = Field(default=True, description="Enable background restoration")
    face_restoration: bool = Field(default=True, description="Enable face restoration")
    batch_input_folder: str = Field(default="/workspace/adjusted", description="Batch input folder")
    batch_output_path: str = Field(default="/workspace/adjustedupscaled", description="Batch output path")
    
    # Advanced options
    edm_steps: int = Field(default=50, ge=1, le=200, description="Number of EDM steps")
    s_cfg: float = Field(default=7.5, ge=1.0, le=30.0, description="CFG scale")
    s_stage1: int = Field(default=-1, description="Stage 1 strength")
    s_stage2: float = Field(default=1.0, description="Stage 2 strength")
    s_churn: int = Field(default=5, description="Churn parameter")
    s_noise: float = Field(default=1.003, description="Noise parameter")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    color_fix_type: str = Field(default="Wavelet", description="Color fix type")
    weight_dtype: str = Field(default="bf16", description="Autoencoder data type")
    diff_dtype: str = Field(default="bf16", description="Diffusion data type")
    sampler: str = Field(default="DPMPP2M", description="Sampler type")
    linear_cfg: bool = Field(default=False, description="Use linear CFG")
    linear_s_stage2: bool = Field(default=False, description="Use linear stage 2")
    face_resolution: int = Field(default=1024, description="Face resolution")
    max_megapixels: float = Field(default=0, description="Maximum megapixels (0 for no limit)")
    max_resolution: int = Field(default=0, description="Maximum resolution (0 for no limit)")

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float = 0.0
    created_at: str
    updated_at: str
    message: str = ""
    error: Optional[str] = None
    result_files: List[str] = []

class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    message: str

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# Helper functions
def create_job_id() -> str:
    """Generate a unique job ID"""
    return str(uuid.uuid4())

def get_job_status(job_id: str) -> Optional[Dict]:
    """Get job status from memory (in production, use Redis/database)"""
    return jobs.get(job_id)

def update_job_status(job_id: str, status: str, progress: float = None, message: str = "", error: str = None, result_files: List[str] = None):
    """Update job status"""
    if job_id not in jobs:
        return
    
    job = jobs[job_id]
    job["status"] = status
    job["updated_at"] = datetime.now().isoformat()
    job["message"] = message
    
    if progress is not None:
        job["progress"] = progress
    if error is not None:
        job["error"] = error
    if result_files is not None:
        job["result_files"] = result_files

def load_supir_model(model_name: str, checkpoint_type: str):
    """Load SUPIR model if not already loaded"""
    global model
    
    if model is None:
        printt(f"Loading SUPIR model: {model_name}")
        try:
            model_path = get_model_path(model_name)
            if not model_path or not os.path.exists(model_path):
                # Try to use get_ckpt_path as fallback
                model_path = get_ckpt_path(model_name)
                if not model_path or not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model not found: {model_name}")
            
            # Try to create SUPIR model with minimal dependencies
            try:
                # Use the full sampler target path instead of just the name
                sampler_target = "sgm.modules.diffusionmodules.sampling.RestoreDPMPP2MSampler"
                
                model = create_SUPIR_model(
                    config_path="options/SUPIR_v0.yaml",
                    supir_sign="Q" if "quality" in model_name.lower() else "F",
                    weight_dtype="bf16",
                    device=SUPIR_device,
                    ckpt=model_path,  # Pass model path directly
                    sampler=sampler_target  # Use full module path
                )
                
                printt("SUPIR model loaded successfully")
            except Exception as e:
                printt(f"Failed to load SUPIR model with create_SUPIR_model: {e}")
                # For now, we'll skip model loading and handle it gracefully
                model = None
                raise e
            
        except Exception as e:
            printt(f"Error loading SUPIR model: {e}")
            raise e

def get_model_path(model_name: str) -> str:
    """Get the full path to a model file"""
    # Check various possible locations
    possible_paths = [
        MODELS_PATH / "checkpoints" / model_name,
        MODELS_PATH / model_name,
        Path("models") / "checkpoints" / model_name,
        Path("models") / model_name
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return ""

def load_llava_agent():
    """Load LLaVA agent if not already loaded"""
    global llava_agent
    
    if llava_agent is None:
        try:
            printt("Loading LLaVA agent")
            llava_agent = LLavaAgent(
                model_path=LLAVA_MODEL_PATH,
                device=LLaVA_device,
                load_8bit=False,
                load_4bit=False
            )
            printt("LLaVA agent loaded successfully")
        except Exception as e:
            printt(f"Error loading LLaVA agent: {e}")
            # Don't raise, just set to None and continue
            llava_agent = None

def load_face_helper():
    """Load face restoration helper if not already loaded"""
    global face_helper
    
    if face_helper is None:
        try:
            printt("Loading face restoration helper")
            face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=SUPIR_device
            )
            printt("Face restoration helper loaded successfully")
        except Exception as e:
            printt(f"Error loading face helper: {e}")
            # Don't raise, just set to None and continue
            face_helper = None


# In your FastAPI file, near the top, add:
from supir_utils import run_supir

# … rest of your imports remain …

async def process_image_job(job_id: str, image_path: str, settings: JobSettings):
    """Process a single image upscaling job using run_supir(...) and save results."""
    try:
        # 1) Mark job as processing
        update_job_status(job_id, "processing", progress=0.1, message="Initializing SUPIR…")

        # 2) Load the uploaded image into a NumPy array
        with Image.open(image_path) as pil:
            pil = pil.convert('RGB')
            image_np = np.array(pil)

        # 3) Extract all needed settings
        checkpoint_name   = settings.model
        checkpoint_type   = settings.checkpoint_type
        upscale_size      = settings.upscale_size
        edm_steps         = settings.edm_steps
        s_cfg             = settings.s_cfg
        s_stage1          = settings.s_stage1
        s_stage2          = settings.s_stage2
        s_churn           = settings.s_churn
        s_noise           = settings.s_noise
        color_fix_type    = settings.color_fix_type
        linear_cfg        = settings.linear_cfg
        linear_s_stage2   = settings.linear_s_stage2
        seed              = settings.seed
        num_samples       = 1
        p_p               = ""  # no additional positive prompt
        n_p               = ""  # no negative prompt
        cfg_scale_start   = 0
        control_scale_start = 0
        apply_face        = settings.face_restoration
        face_prompt       = settings.prompt if settings.prompt else ""
        face_resolution   = settings.face_resolution
        max_megapixels    = settings.max_megapixels
        max_resolution    = settings.max_resolution

        update_job_status(job_id, "processing", progress=0.2, message="Running SUPIR upscaling…")

        # 4) Call run_supir(...)
        try:
            output_np = run_supir(
                image_np,
                checkpoint_name=checkpoint_name,
                checkpoint_type=checkpoint_type,
                upscale_size=upscale_size,
                edm_steps=edm_steps,
                s_cfg=s_cfg,
                s_stage1=s_stage1,
                s_stage2=s_stage2,
                s_churn=s_churn,
                s_noise=s_noise,
                color_fix_type=color_fix_type,
                linear_cfg=linear_cfg,
                linear_s_stage2=linear_s_stage2,
                seed=seed,
                num_samples=num_samples,
                p_p=p_p,
                n_p=n_p,
                cfg_scale_start=cfg_scale_start,
                control_scale_start=control_scale_start,
                apply_face=apply_face,
                face_prompt=face_prompt,
                face_resolution=face_resolution,
                max_megapixels=max_megapixels,
                max_resolution=max_resolution
            )
            update_job_status(job_id, "processing", progress=0.7, message="SUPIR upscaling completed")
        except Exception as supir_err:
            # If run_supir itself raises (unlikely, since it already falls back),
            # we catch it here and do a simple bicubic fallback.
            printt(f"run_supir error: {supir_err}")
            update_job_status(job_id, "processing", progress=0.7,
                              message="SUPIR failed, using fallback bicubic…")
            pil_fallback = Image.fromarray(image_np).resize(
                (int(image_np.shape[1] * upscale_size), int(image_np.shape[0] * upscale_size)),
                resample=Image.BICUBIC
            )
            output_np = np.array(pil_fallback)

        # 5) Save result(s) to disk
        update_job_status(job_id, "processing", progress=0.8, message="Saving results…")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_fn = f"{job_id}_{timestamp}"
        result_files = []

        # Primary output PNG
        out_filename = f"{base_fn}.png"
        out_path = OUTPUT_FOLDER / out_filename
        Image.fromarray(output_np).save(out_path, "PNG")
        result_files.append(str(out_path))

        # If user wants captions, write a .txt with the prompt
        if settings.save_captions and settings.prompt:
            cap_fn = f"{base_fn}_caption.txt"
            cap_path = OUTPUT_FOLDER / cap_fn
            with open(cap_path, 'w', encoding='utf-8') as f:
                f.write(settings.prompt)
            result_files.append(str(cap_path))

        # 6) Mark job complete
        update_job_status(
            job_id,
            "completed",
            progress=1.0,
            message="Processing completed successfully",
            result_files=result_files
        )

        # 7) Clean up the temp file if it lives in TEMP_UPLOAD_PATH
        if os.path.exists(image_path) and str(TEMP_UPLOAD_PATH) in image_path:
            os.remove(image_path)

        printt(f"Job {job_id} completed, outputs: {result_files}")

    except Exception as e:
        # Catch‐all: mark as failed and clean up
        err_msg = f"{type(e).__name__}: {e}"
        printt(f"Job {job_id} failed: {err_msg}")
        traceback.print_exc()
        update_job_status(job_id, "failed", error=err_msg)
        if os.path.exists(image_path) and str(TEMP_UPLOAD_PATH) in image_path:
            os.remove(image_path)


# API Endpoints

@app.post("/job", response_model=JobCreateResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Image file to upscale"),
    settings: str = Form(default='{}', description="Job settings as JSON string"),
    token: str = Depends(verify_token)
):
    """
    Create a new upscaling job
    
    - **image**: Image file (JPEG, PNG, WebP, etc.)
    - **settings**: JSON string with processing settings
    """
    try:
        # Parse settings
        try:
            settings_dict = json.loads(settings) if settings != '{}' else {}
            job_settings = JobSettings(**settings_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid settings: {str(e)}")
        
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create job ID
        job_id = create_job_id()
        
        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(image.filename).suffix if image.filename else '.png'
        temp_filename = f"{job_id}_{timestamp}{file_extension}"
        temp_path = TEMP_UPLOAD_PATH / temp_filename
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Create job record
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message": "Job created, queued for processing",
            "error": None,
            "result_files": [],
            "settings": job_settings.dict(),
            "image_path": str(temp_path)
        }
        
        # Start background processing
        background_tasks.add_task(process_image_job, job_id, str(temp_path), job_settings)
        
        return JobCreateResponse(
            job_id=job_id,
            status="pending",
            message="Job created successfully and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status_endpoint(job_id: str, token: str = Depends(verify_token)):
    """
    Get the status of a specific job
    
    - **job_id**: The ID of the job to check
    """
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**job)

@app.get("/job/{job_id}/result")
async def get_job_result(job_id: str, file_index: int = 0, token: str = Depends(verify_token)):
    """
    Download the result of a completed job
    
    - **job_id**: The ID of the completed job
    - **file_index**: Index of the result file to download (default: 0)
    """
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed. Current status: {job['status']}")
    
    if not job["result_files"]:
        raise HTTPException(status_code=404, detail="No result files found")
    
    if file_index >= len(job["result_files"]):
        raise HTTPException(status_code=400, detail=f"File index {file_index} out of range. Available files: {len(job['result_files'])}")
    
    result_file = job["result_files"][file_index]
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Result file not found on disk")
    
    filename = Path(result_file).name
    return FileResponse(
        result_file,
        media_type="image/png",
        filename=filename
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "models_loaded": {
            "supir": model is not None,
            "llava": llava_agent is not None,
            "face_helper": face_helper is not None
        }
    }

@app.get("/jobs")
async def list_jobs(token: str = Depends(verify_token)):
    """List all jobs (for debugging)"""
    return {"jobs": list(jobs.keys()), "total": len(jobs)}

@app.delete("/job/{job_id}")
async def delete_job(job_id: str, token: str = Depends(verify_token)):
    """Delete a job and its results"""
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up result files
    for result_file in job.get("result_files", []):
        if os.path.exists(result_file):
            os.remove(result_file)
    
    # Remove from jobs dict
    del jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

# if __name__ == "__main__":
#     import uvicorn
    
#     # Get configuration from environment
#     host = os.getenv("HOST", "0.0.0.0")
#     port = int(os.getenv("PORT", "8000"))
    
#     print(f"Starting SUPIR API server on {host}:{port}")
#     print(f"API Token: {API_TOKEN}")
#     print(f"SUPIR Device: {SUPIR_device}")
#     print(f"LLaVA Device: {LLaVA_device}")
    
#     uvicorn.run(app, host=host, port=port) s