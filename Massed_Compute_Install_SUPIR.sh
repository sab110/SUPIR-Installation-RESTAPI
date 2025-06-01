pip install requests

git clone https://github.com/FurkanGozukara/SUPIR

cd SUPIR

python3 -m venv venv

source ./venv/bin/activate

pip install -r requirements.txt

echo "Installing requirements"

pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade
pip install triton
pip install xformers==0.0.24
pip install bitsandbytes==0.43.0 --upgrade

cd ..

pip install ipywidgets
pip install hf_transfer huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 HF_model_downloader.py

echo "Virtual environment made and installed properly"

read -p "Press [Enter] key to continue..."
