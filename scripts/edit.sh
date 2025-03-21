eval "$(conda shell.bash hook)"
conda activate opengs
export CUDA_VISIBLE_DEVICES=4
# python render_lerf_by_text.py -m "./output/IMG8304" --scene_name "IMG8304"
python render_lerf_by_text.py -m "./output/0.3" --scene_name "figurines"