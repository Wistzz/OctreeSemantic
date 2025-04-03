eval "$(conda shell.bash hook)"
conda activate opengs
export CUDA_VISIBLE_DEVICES=4



# python render_lerf_by_text.py -m "./output/f57605cd-9" --scene_name "teatime"
python render_lerf_by_text.py -m "./output/kitchen200" --scene_name "waldo_kitchen"
python render_lerf_by_text.py -m "./output/kitchen400" --scene_name "waldo_kitchen"
python render_lerf_by_text.py -m "./output/kitchen600" --scene_name "waldo_kitchen"
python render_lerf_by_text.py -m "./output/kitchen800" --scene_name "waldo_kitchen"
python render_lerf_by_text.py -m "./output/kitchen1000" --scene_name "waldo_kitchen"
python render_lerf_by_text.py -m "./output/kitchen1200" --scene_name "waldo_kitchen"
python render_lerf_by_text.py -m "./output/kitchen1400" --scene_name "waldo_kitchen"
python render_lerf_by_text.py -m "./output/kitchen1600" --scene_name "waldo_kitchen"

# python render_lerf_by_text.py -m "./output/73edf569-1" --scene_name "ramen"
# python render_lerf_by_text.py -m "./output/figurines_detach_sota" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/teatime_detach_sota" --scene_name "teatime"
# python render_lerf_by_text.py -m "./output/kitchen_detach_sota" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/ramen_detach_sota" --scene_name "ramen"

# python render_lerf_by_text.py -m "./output/teatime100" --scene_name "teatime"
# python render_lerf_by_text.py -m "./output/0.015" --scene_name "figurines"

