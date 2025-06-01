pip install requests

# git clone https://github.com/FurkanGozukara/SUPIR

cd SUPIR

rm -rf ~/.cache/pip

python -m venv venv

source ./venv/bin/activate

pip install antlr4-python3-runtime==4.9.3
pip install openai-clip==1.0.1
pip install filterpy==1.4.5

pip install -r requirements.txt

echo "Installing requirements"

pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade
pip install triton
pip install xformers==0.0.28.post.1
pip install bitsandbytes==0.43.1 --upgrade

cd ..

pip install ipywidgets
pip install hf_transfer huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1
python HF_model_downloader.py

echo "Virtual environment made and installed properly"

read -p "Press [Enter] key to continue..."
