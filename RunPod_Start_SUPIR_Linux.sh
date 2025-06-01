cd SUPIR

source ./venv/bin/activate
export PYTHONWARNINGS=ignore
# Uncomment and set CUDA_VISIBLE_DEVICES=0 to use a specific CUDA device
# export CUDA_VISIBLE_DEVICES=0

echo "Use both VRAM optimizations and it uses around 12GB VRAM GPU"
echo "FP8 optimization uses around 8GB VRAM"
echo "If you have over 30 GB VRAM, you can start both full Params and no Tiled-VAE"
echo "Modify this file and remove --share if you don't want Gradio share"

echo
echo

echo "Please select an option:"
echo "1. Start As BF16/FP16 - Uses Lesser VRAM Preferred"
echo "2. Start As FP8 - Uses More Lesser VRAM - Very Good For 12GB and Below GPUs"
echo "3. Start As Full Precision"
echo ""

read -p "Enter your choice (1-3): " choice

echo "Please select an option:"
echo "1. Start Using Tiled VAE - Uses Lesser VRAM Preferred"
echo "2. Start Without Tiled VAE"
echo ""

read -p "Enter your choice (1-2): " choice2

echo "Please select an option:"
echo "1. Start With Auto Move To CPU - Useful When Using LLaVA - Moves model to CPU to save VRAM for LLaVA"
echo "2. Keep Models Always In GPU"
echo ""

read -p "Enter your choice (1-2): " choice3

echo "Please select an option:"
echo "1. Start With Light Theme"
echo "2. Start With Dark Theme"
echo ""

read -p "Enter your choice (1-2): " choice4


lowvram=""
tiledVAE=""
cpuMove=""
theme=""

if [ "$choice" == "1" ]; then
    lowvram="--loading_half_params"
fi
if [ "$choice" == "2" ]; then
    lowvram="--loading_half_params --fp8"
fi
if [ "$choice2" == "1" ]; then
    tiledVAE="--use_tile_vae"
fi
if [ "$choice3" == "2" ]; then
    cpuMove="--dont_move_cpu"
fi
if [ "$choice4" == "2" ]; then
    theme="--theme d8ahazard/material_design_rd"
fi

python gradio_demo.py $lowvram $tiledVAE $theme $cpuMove --open_browser --share True

# Uncomment and modify the following lines as needed
# python gradio_demo.py $lowvram $tiledVAE $slider $theme --ckpt "R:/SUPIR_v8/SUPIR/models/sd_xl_base_1.0_0.9vae.safetensors"
# python gradio_demo.py $lowvram $tiledVAE $slider $theme --ckpt "R:/SUPIR_v8/SUPIR/models/sd_xl_base_1.0_0.9vae.safetensors"
# python gradio_demo.py $lowvram $tiledVAE --outputs_folder "R:/SUPIR_v8/test2"

read -p "Press any key to continue . . . " -n1 -s
