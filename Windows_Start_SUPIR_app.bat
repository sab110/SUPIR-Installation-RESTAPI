@echo off

cd SUPIR

call .\venv\Scripts\activate.bat || exit /b
REM SET CUDA_VISIBLE_DEVICES=0  - this is used to set certain CUDA device visible only used
set PYTHONWARNINGS=ignore
REM SET CUDA_VISIBLE_DEVICES=1
echo Use both VRAM optimizations and it uses around 12GB VRAM GPU
echo FP8 optimization uses around 8GB VRAM
echo If you have over 30 GB VRAM, you can start both full Params and no Tiled-VAE
echo Modify this file and add --share if you want Gradio share

echo.
echo.

echo Please select an option:
echo 1. Start As BF16/FP16 - Uses Lesser VRAM Preferred
echo 2. Start As FP8 - Uses More Lesser VRAM - Very Good For 12GB and Below GPUs
echo 3. Start As Full Precision
echo.

set /p choice="Enter your choice (1-3): "

echo Please select an option:
echo 1. Start Using Tiled VAE - Uses Lesser VRAM Preferred
echo 2. Start Without Tiled VAE
echo.

set /p choice2="Enter your choice (1-2): "

echo Please select an option:
echo 1. Start With Auto Move To CPU - Useful When Using LLaVA - Moves model to CPU to save VRAM for LLaVA
echo 2. Keep Models Always In GPU
echo.

set /p choice3="Enter your choice (1-2): "

echo Please select an option:
echo 1. Start With Light Theme
echo 2. Start With Dark Theme
echo.

set /p choice4="Enter your choice (1-2): "


set "lowvram="
set "tiledVAE="
set "cpuMove="
set "theme="

if "%choice%" == "1" (
    set "lowvram=--loading_half_params"
)
if "%choice%" == "2" (
    set "lowvram=--loading_half_params --fp8"
)
if "%choice2%" == "1" (
    set "tiledVAE=--use_tile_vae"
)
if "%choice3%" == "2" (
    set "cpuMove=--dont_move_cpu"
)
if "%choice4%" == "2" (
    set "theme=--theme d8ahazard/material_design_rd"
)

python gradio_demo.py %lowvram% %tiledVAE% %theme% %cpuMove% --open_browser --outputs_folder_button

REM python gradio_demo.py %lowvram% %tiledVAE% %slider% %theme% --ckpt "R:\SUPIR_v8\SUPIR\models\sd_xl_base_1.0_0.9vae.safetensors"

REM example SDXL model path python gradio_demo.py %lowvram% %tiledVAE% %slider% %theme% --ckpt "R:\SUPIR_v8\SUPIR\models\sd_xl_base_1.0_0.9vae.safetensors"

REM example folder path python gradio_demo.py %lowvram% %tiledVAE% --outputs_folder "R:\SUPIR_v8\test2"

pause
