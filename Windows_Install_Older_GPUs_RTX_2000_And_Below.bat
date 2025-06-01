echo WARNING. For this auto installer to work you need to have installed Python 3.10.11 or 3.10.13 and C++ tools 
echo follow this tutorial : https://youtu.be/-NjNy7afOQ0

pip install requests
pip install tqdm

git clone https://github.com/FurkanGozukara/SUPIR

cd SUPIR

py --version >nul 2>&1
if "%ERRORLEVEL%" == "0" (
    echo Python launcher is available. Generating Python 3.10 VENV
    py -3.10 -m venv venv
) else (
    echo Python launcher is not available, generating VENV with default Python. Make sure that it is 3.10
    python -m venv venv
)

call .\venv\Scripts\activate.bat

pip install tqdm

pip install -r requirements.txt

echo installing requirements 

pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp310-cp310-win_amd64.whl

pip install bitsandbytes==0.43.3 --upgrade

pip uninstall torch torchvision torchaudio xformers --yes
pip3 install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121

cd ..


pip install ipywidgets
pip install hf_transfer
set HF_HUB_ENABLE_HF_TRANSFER=1
python HF_model_downloader.py

REM Show completion message
echo Virtual environment made and installed properly

REM Pause to keep the command prompt open
pause