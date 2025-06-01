cd SUPIR

call .\venv\Scripts\activate.bat || exit /b

python gradio_demo.py --loading_half_params --fp8 --use_tile_vae --open_browser --outputs_folder_button

pause
