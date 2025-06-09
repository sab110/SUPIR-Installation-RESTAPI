# SUPIR - Super-Resolution AI with REST API

A powerful super-resolution AI application based on SUPIR technology, designed for easy deployment on RunPod with both Gradio UI and REST API endpoints.

**Repository**: https://github.com/sab110/SUPIR-Installation-RESTAPI.git

## 🎯 Quick Start on RunPod

### Minimum Requirements

#### GPU Requirements
- **Minimum**: 24GB VRAM (RTX 3090, RTX 4090)  
- **Recommended**: 48GB VRAM (A40, A6000, L40S) for maximum speed and no-tiling
- **Budget Option**: 12GB VRAM possible with FP8 precision (will be slower)

#### Storage
- **Minimum**: 100GB disk space
- **Recommended**: 150GB+ for model storage and processing (models are ~50GB)

### RunPod Setup Instructions

#### 1. Register and Setup
1. Register for RunPod at: https://runpod.io?ref=1aka98lq
2. Watch setup tutorial: https://youtu.be/KW-MHmoNcqo?si=QN8X8Sjn13ZYu-EU&t=1323 (starts at 22:03)
3. For persistent storage: https://youtu.be/8Qf4x3-DFf4

#### 2. Pod Configuration
1. **Select GPU**: Choose minimum 24GB VRAM
   - RTX 3090/4090: Good for most use cases
   - A40/A6000/L40S: Best performance for professional use
2. **Template**: Select `RunPod Pytorch 2.2.0`
   ```
   runpod/pytorch:2.2.0-py3.10-cuda12.1.1
   ```
3. **Storage**: Set volume disk to 100GB or bigger
4. **Ports**: Add HTTP port 8000 for FastAPI access

#### 3. Installation
1. Clone the repository to `/workspace`:
   ```bash
   cd /workspace
   git clone https://github.com/sab110/SUPIR-Installation-RESTAPI.git .
   ```
   
   Or upload the project files to `/workspace` folder (you can upload as zip and extract)

2. Set environment and run installation:
   ```bash
   export HF_HOME="/workspace"
   chmod +x RunPod_Install_SUPIR.sh
   ./RunPod_Install_SUPIR.sh
   ```
3. If model downloads fail, re-run the installation command until all models download properly

#### 4. Starting the FastAPI Server

##### Option A: Using the Entrypoint Script (Recommended)
```bash
cd /workspace
chmod +x entrypoint.sh
./entrypoint.sh
```

This script will:
- Activate the virtual environment
- Download models if needed (~50GB)
- Copy models to persistent storage
- Start the FastAPI server on port 8000

##### Option B: Manual FastAPI Server Start
```bash
cd /workspace
export HF_HOME="/workspace"
export PYTHONWARNINGS=ignore
apt update && apt install ffmpeg --yes
chmod +x RunPod_Start_SUPIR_Linux.sh
./RunPod_Start_SUPIR_Linux.sh
```

**Access the API**: `http://your-pod-ip:8000`

##### Option C: Gradio Interface (Alternative)
```bash
cd /workspace/SUPIR
source ./venv/bin/activate
export PYTHONWARNINGS=ignore
python gradio_demo.py --loading_half_params --use_tile_vae --share
```
Access UI via Gradio share URL or `http://your-pod-ip:7861`

## 🔧 Configuration Options

### Memory Optimization Settings

| Setting | VRAM Usage | Best For |
|---------|------------|----------|
| `--loading_half_params` | ~12GB | 24GB GPUs (RTX 3090/4090) |
| `--loading_half_params --fp8` | ~8GB | 12GB GPUs |
| No flags | ~20GB+ | 48GB+ professional GPUs |

### Common Command Line Arguments

```bash
# Memory optimizations
--loading_half_params    # Use FP16 precision (saves VRAM)
--fp8                   # Use FP8 precision (saves more VRAM)
--use_tile_vae          # Enable tiling for large images
--dont_move_cpu         # Keep models in GPU (if you have enough VRAM)

# Server options
--share                 # Enable Gradio sharing
--port 7861            # Set custom port
--ip 0.0.0.0           # Bind to all interfaces

# Quality options
--fast_load_sd         # Faster loading (may use more VRAM)
--encoder_tile_size 512 # Custom tile size for encoder
--decoder_tile_size 64  # Custom tile size for decoder
```

## 📚 GPU Performance Guide

### RTX 3090/4090 (24GB)
```bash
python gradio_demo.py --loading_half_params --use_tile_vae --share
```
- **Processing Time**: ~30-60 seconds per image
- **Max Resolution**: 2K-4K with tiling
- **Best Settings**: FP16 + Tiled VAE

### RTX 4080/4070 (12-16GB)
```bash
python gradio_demo.py --loading_half_params --fp8 --use_tile_vae --share
```
- **Processing Time**: ~60-120 seconds per image
- **Max Resolution**: 1K-2K with tiling
- **Best Settings**: FP8 + Tiled VAE

### Professional GPUs (A40/A6000/L40S - 48GB)
```bash
python gradio_demo.py --dont_move_cpu --share
```
- **Processing Time**: ~15-30 seconds per image
- **Max Resolution**: 8K+ without tiling
- **Best Settings**: Full precision, no tiling

## 🌐 FastAPI Server Details

The FastAPI server (`api_server.py`) runs on port 8000 and provides RESTful endpoints for super-resolution processing.

### Server Startup Process
1. **Environment Setup**: Activates virtual environment and sets Python warnings
2. **Model Loading**: Loads SUPIR models with memory optimizations
3. **API Binding**: Binds to `0.0.0.0:8000` for external access
4. **VRAM Optimization**: Automatically configures FP16/FP8 based on available VRAM

### Main API Endpoints
- `POST /upscale` - Main super-resolution endpoint
- `GET /health` - Health check
- `GET /models` - List available models
- `GET /docs` - FastAPI documentation (Swagger UI)

### Example API Usage
```python
import requests

# Health check
response = requests.get('http://your-pod-ip:8000/health')
print(response.json())

# Upscale image
files = {'image': open('input.jpg', 'rb')}
data = {
    'scale_factor': 4,
    'quality': 'high'
}

response = requests.post('http://your-pod-ip:8000/upscale', files=files, data=data)
with open('output.jpg', 'wb') as f:
    f.write(response.content)
```

## 🚀 Optimization Tips

### For Best Performance
1. Use professional GPUs (48GB) when possible
2. Enable `--dont_move_cpu` if you have sufficient VRAM
3. Use NVMe storage for faster model loading
4. Consider batch processing for multiple images

### For Limited VRAM
1. Always use `--loading_half_params --fp8`
2. Enable `--use_tile_vae` for large images
3. Process images sequentially, not in batch
4. Consider lower resolution inputs

### Troubleshooting
1. **Out of Memory**: Add `--fp8` and `--use_tile_vae`
2. **Slow Loading**: Ensure models are downloaded to `/workspace`
3. **Connection Issues**: Check if port 8000 is accessible
4. **Model Download Fails**: Re-run installation script
5. **API Not Starting**: Check if virtual environment is activated and dependencies installed

## 📁 Project Structure

```
├── SUPIR/                 # Main application directory
│   ├── gradio_demo.py     # Gradio web interface
│   ├── api_server.py      # FastAPI REST API server
│   ├── requirements.txt   # Python dependencies
│   └── models/           # Model storage directory
├── RunPod_Install_SUPIR.sh    # Installation script
├── RunPod_Start_SUPIR_Linux.sh # FastAPI server startup script
├── entrypoint.sh          # Docker/automated startup script
├── HF_model_downloader.py # Model download utility
└── README.md             # This documentation
```

## 🔗 Useful Links

- **GitHub Repository**: https://github.com/sab110/SUPIR-Installation-RESTAPI.git
- **RunPod Registration**: https://runpod.io?ref=1aka98lq
- **Setup Tutorial**: https://youtu.be/KW-MHmoNcqo?t=1323
- **Storage Tutorial**: https://youtu.be/8Qf4x3-DFf4

## 📝 License

This project includes modifications and enhancements to the original SUPIR codebase. Please refer to the LICENSE file for details.

---

**Happy Super-Resolution Processing! 🚀** 