#!/usr/bin/env bash
set -e

# 1. Activate the SUPIR venv
cd /workspace/SUPIR
source venv/bin/activate

# 2. Make sure /workspace/models exists (persisted by RunPod)
mkdir -p /workspace/models

# 3. If SUPIR/models is empty, run the default HF_model_downloader.py
#    (This will pull ~50 GB into SUPIR/models/)
if [ -z "$(ls -A /workspace/SUPIR/models 2>/dev/null)" ]; then
    echo ">>> SUPIR/models is empty—downloading weights (~50 GB) now..."
    python HF_model_downloader.py
    echo ">>> Model download complete."
else
    echo ">>> SUPIR/models already has files—skipping download."
fi

# 4. Copy all weights from SUPIR/models/ → /workspace/models/
echo ">>> Copying SUPIR/models/* → /workspace/models/"
cp -r /workspace/SUPIR/models/* /workspace/models/

# 5. Finally, launch the FastAPI server via your existing start script
#    (It does: cd SUPIR; source venv/bin/activate; uvicorn api_server:app)
echo ">>> Starting SUPIR API server…"
exec bash /workspace/RunPod_Start_SUPIR_Linux.sh
