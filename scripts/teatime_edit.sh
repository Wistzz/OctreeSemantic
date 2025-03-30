eval "$(conda shell.bash hook)"
conda activate opengs
export CUDA_VISIBLE_DEVICES=3

python render_lerf_by_text.py -m "./output/0" --scene_name "ramen"
python render_lerf_by_text.py -m "./output/0.1" --scene_name "ramen"
python render_lerf_by_text.py -m "./output/0.01" --scene_name "ramen"
python render_lerf_by_text.py -m "./output/0.2" --scene_name "ramen"
python render_lerf_by_text.py -m "./output/0.5" --scene_name "ramen"
python render_lerf_by_text.py -m "./output/0.05" --scene_name "ramen"
python render_lerf_by_text.py -m "./output/1" --scene_name "ramen"
python render_lerf_by_text.py -m "./output/0.02" --scene_name "ramen"

# python render_lerf_by_text.py -m "./output/teatime100" --scene_name "teatime"
# python render_lerf_by_text.py -m "./output/0.015" --scene_name "figurines"

