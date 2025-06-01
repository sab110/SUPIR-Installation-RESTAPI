echo WARNING. For this auto installer to work you need to have installed Python 3.10.11 or 3.10.13 and C++ tools 
echo follow this tutorial : https://youtu.be/-NjNy7afOQ0

pip install requests
pip install tqdm

cd SUPIR

git pull

call .\venv\Scripts\activate.bat

python -m pip install --upgrade pip

pip install tqdm

pip install -r requirements.txt

pip install bitsandbytes --upgrade

pip install --pre torch==2.7.0.dev20250311 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

REM pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/flash_attn-2.7.4.post1-cp310-cp310-win_amd64.whl

pip install https://files.pythonhosted.org/packages/3d/65/1a6394f5d6dee851e9ea59e385f6d6428e3bfe36f83c06e0336e14dcfd11/deepspeed-0.16.4-cp310-cp310-win_amd64.whl

pip install -U --pre triton-windows

pip install torchao

pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/xformers-0.0.30+836cd905.d20250327-cp310-cp310-win_amd64.whl

cd ..


pip install ipywidgets
pip install hf_transfer
set HF_HUB_ENABLE_HF_TRANSFER=1
python HF_model_downloader.py

REM Show completion message
echo Virtual environment made and installed properly

REM Pause to keep the command prompt open
pause