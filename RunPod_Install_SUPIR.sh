# pip install requests

# # git clone https://github.com/FurkanGozukara/SUPIR

# cd SUPIR

# rm -rf ~/.cache/pip

# python -m venv venv

# source ./venv/bin/activate

# pip install antlr4-python3-runtime==4.9.3
# pip install openai-clip==1.0.1
# pip install filterpy==1.4.5

# pip install -r requirements.txt

# echo "Installing requirements"

# pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade
# pip install triton
# pip install xformers==0.0.28.post.1
# pip install bitsandbytes==0.43.1 --upgrade

# cd ..

# pip install ipywidgets
# pip install hf_transfer huggingface_hub
# export HF_HUB_ENABLE_HF_TRANSFER=1
# python HF_model_downloader.py

# echo "Virtual environment made and installed properly"

# read -p "Press [Enter] key to continue..."

# ----------------------------------------
# 1. Install system-level prerequisites (if not already installed)
# ----------------------------------------
pip install requests

# ----------------------------------------
# 2. Enter the existing SUPIR directory
# ----------------------------------------
cd SUPIR

# ----------------------------------------
# 3. Clean pip cache (optional, to force fresh installs)
# ----------------------------------------
rm -rf ~/.cache/pip

# ----------------------------------------
# 4. Create and activate a Python virtual environment
# ----------------------------------------
python3 -m venv venv
source ./venv/bin/activate

# ----------------------------------------
# 5. Install low-level dependencies
# ----------------------------------------
pip install --upgrade pip setuptools wheel
pip install antlr4-python3-runtime==4.9.3
pip install openai-clip==1.0.1
pip install filterpy==1.4.5

# ----------------------------------------
# 6. Install the requirements.txt
# ----------------------------------------
pip install -r requirements.txt

echo "Installing requirements"

# ----------------------------------------
# 7. Install PyTorch + related libraries
# ----------------------------------------
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade
pip install triton
pip install xformers==0.0.28.post.1
pip install bitsandbytes==0.43.1 --upgrade

# ----------------------------------------
# 8. Install REST-API–specific dependencies
# ----------------------------------------
pip install fastapi uvicorn[standard] python-multipart
pip install pydantic pillow numpy

# ----------------------------------------
# 9. Install HuggingFace-related tools
# ----------------------------------------
pip install ipywidgets
pip install hf_transfer huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1

# ----------------------------------------
# 10. Download any HF models via HF_model_downloader.py
# ----------------------------------------
if [ -f "HF_model_downloader.py" ]; then
    python HF_model_downloader.py
else
    echo "Warning: HF_model_downloader.py not found; skipping."
fi

echo "Virtual environment made and installed properly"

read -p "Press [Enter] key to continue..."
