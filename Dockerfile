# --------------------------------------------------------------------------------
# 1. Base image – PyTorch 2.1 with CUDA 11.8 (matches RunPod PyTorch 2.1 runtime)
# --------------------------------------------------------------------------------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --------------------------------------------------------------------------------
# 2. Install OS‐level prerequisites
# --------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        libgl1-mesa-glx \
        ca-certificates \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------------
# 3. Create /workspace and clone your SUPIR‐Installation‐RESTAPI repo
# --------------------------------------------------------------------------------
WORKDIR /workspace

# Clone into “.” so that /workspace now contains:
# ├─ RunPod_Install_SUPIR.sh
# ├─ RunPod_Start_SUPIR_Linux.sh
# ├─ SUPIR/           (folder with all SUPIR code)
# ├─ requirements.txt, api_server.py, supir_utils.py, etc.
RUN git clone https://github.com/sab110/SUPIR-Installation-RESTAPI.git . 

# --------------------------------------------------------------------------------
# 4. Make both installer and startup scripts executable
# --------------------------------------------------------------------------------
RUN chmod +x RunPod_Install_SUPIR.sh RunPod_Start_SUPIR_Linux.sh

# --------------------------------------------------------------------------------
# 5. Run the installer script (non‐interactive)
#    This script should:
#      • cd into SUPIR/
#      • create a virtualenv at SUPIR/venv
#      • pip‐install all prerequisites (PyTorch, xformers, bitsandbytes, FastAPI, etc.)
#      • run HF_model_downloader.py (if present) to pull any models
# --------------------------------------------------------------------------------
RUN bash RunPod_Install_SUPIR.sh

# --------------------------------------------------------------------------------
# 6. Copy any downloaded model files from SUPIR/models → /workspace/models
#    so that supir_config.py can locate them under /workspace/models/…
# --------------------------------------------------------------------------------
# RUN mkdir -p /workspace/models \
#  && if [ -d "/workspace/SUPIR/models" ]; then \
#        cp -r /workspace/SUPIR/models/* /workspace/models/ ; \
#     fi

# --------------------------------------------------------------------------------
# 7. Create the “workspace” folders that api_server.py (supir_config.py) expects:
#      • /workspace/adjusted
#      • /workspace/adjustedupscaled
#      • /workspace/models    (already created above)
#      • /workspace/temp
# --------------------------------------------------------------------------------
# RUN mkdir -p /workspace/adjusted \
#              /workspace/adjustedupscaled \
#              /workspace/temp

# --------------------------------------------------------------------------------
# 8. Expose port 8000 so RunPod can forward it to the outside world
# --------------------------------------------------------------------------------
EXPOSE 8000

# --------------------------------------------------------------------------------
# 9. ENTRYPOINT: run the “start” script. This script:
#      • cd into SUPIR/
#      • source the venv
#      • set PYTHONWARNINGS=ignore
#      • launch Uvicorn serving api_server:app on 0.0.0.0:8000
# 
#    Because Uvicorn runs in the foreground, Docker (and RunPod) will keep this
#    container alive. If you want to pin SUPIR to a single GPU (e.g. GPU 0), you can
#    uncomment the CUDA_VISIBLE_DEVICES line below.
# --------------------------------------------------------------------------------
ENTRYPOINT ["/bin/bash", "-c", "\
    cd /workspace/SUPIR && \
    source venv/bin/activate && \
    export PYTHONWARNINGS='ignore' && \
    # (Optional) Pin to GPU 0 only; uncomment if desired:
    # export CUDA_VISIBLE_DEVICES=0 && \
    echo '🚀 Starting SUPIR FastAPI server on 0.0.0.0:8000 ...' && \
    uvicorn api_server:app --host 0.0.0.0 --port 8000 \
"]
