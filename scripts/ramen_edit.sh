eval "$(conda shell.bash hook)"
conda activate opengs
export CUDA_VISIBLE_DEVICES=0
# python render_lerf_by_text.py -m "./output/kitchen_sparse_base" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen_sparse_infonce" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/ramen50" --scene_name "ramen"
# python render_lerf_by_text.py -m "./output/ramen100" --scene_name "ramen"
# python render_lerf_by_text.py -m "./output/ramen200" --scene_name "ramen"
python render_lerf_by_text.py -m "./output/ramen200" --scene_name "ramen"
# python render_lerf_by_text.py -m "./output/ramen600" --scene_name "ramen"
# python render_lerf_by_text.py -m "./output/ramen800" --scene_name "ramen"
# python render_lerf_by_text.py -m "./output/ramen1000" --scene_name "ramen"
# python render_lerf_by_text.py -m "./output/ramen1200" --scene_name "ramen"
