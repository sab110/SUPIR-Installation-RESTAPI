# --------------------------------------------------------------------------------
# 1. Base image – PyTorch 2.1 with CUDA 11.8 (matches RunPod’s runtime)
# --------------------------------------------------------------------------------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Avoid .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --------------------------------------------------------------------------------
# 2. Install OS‐level dependencies
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
# 3. Create /workspace and clone your SUPIR repo into it
# --------------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone https://github.com/sab110/SUPIR-Installation-RESTAPI.git .

# --------------------------------------------------------------------------------
# 4. Make both RunPod scripts and entrypoint.sh executable
# --------------------------------------------------------------------------------
RUN chmod +x RunPod_Install_SUPIR.sh RunPod_Start_SUPIR_Linux.sh

# --------------------------------------------------------------------------------
# 5. Patch out the HF_model_downloader.py call in installation
#
#    We comment out the line "python HF_model_downloader.py" so that
#    no 50 GB is downloaded at build time. That download will happen
#    later in entrypoint.sh instead.
# --------------------------------------------------------------------------------
RUN sed -i "s|python HF_model_downloader.py|# python HF_model_downloader.py|" RunPod_Install_SUPIR.sh

# --------------------------------------------------------------------------------
# 6. Run the installer script to set up venv & pip packages
#    (No HF model download here—only code + dependencies)
# --------------------------------------------------------------------------------
RUN bash RunPod_Install_SUPIR.sh

# --------------------------------------------------------------------------------
# 7. Create persistent "workspace" folders that supir_config.py expects:
#      • /workspace/models           (for large weights later)
#      • /workspace/adjusted
#      • /workspace/adjustedupscaled
#      • /workspace/temp
# --------------------------------------------------------------------------------
RUN mkdir -p /workspace/models \
             /workspace/adjusted \
             /workspace/adjustedupscaled \
             /workspace/temp

# --------------------------------------------------------------------------------
# 8. Expose the FastAPI port for Uvicorn
# --------------------------------------------------------------------------------
EXPOSE 8000

# --------------------------------------------------------------------------------
# 9. Copy entrypoint.sh into the image and make it executable
# --------------------------------------------------------------------------------
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# --------------------------------------------------------------------------------
# 10. At runtime, entrypoint.sh will:
#      1) Activate the venv
#      2) Download weights into SUPIR/models (if not present)
#      3) Copy SUPIR/models/* → /workspace/models/*
#      4) Launch the Uvicorn server via RunPod_Start_SUPIR_Linux.sh
# --------------------------------------------------------------------------------
ENTRYPOINT ["/workspace/entrypoint.sh"]
