eval "$(conda shell.bash hook)"
conda activate opengs
export CUDA_VISIBLE_DEVICES=4
python render_lerf_by_text.py -m "./output/8615d4fc-d" --scene_name "figurines"