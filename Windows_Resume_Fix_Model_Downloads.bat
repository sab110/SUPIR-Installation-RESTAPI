cd SUPIR

call .\venv\Scripts\activate.bat || exit /b

cd ..

pip install huggingface_hub
pip install ipywidgets
pip install hf_transfer
set HF_HUB_ENABLE_HF_TRANSFER=1
python HF_model_downloader.py

pause
