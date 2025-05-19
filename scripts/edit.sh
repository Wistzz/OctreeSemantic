eval "$(conda shell.bash hook)"
conda activate opengs
export CUDA_VISIBLE_DEVICES=1
# python render_lerf_by_text.py -m "./output/kitchen_sparse_base" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen_sparse_infonce" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen400" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen800" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen1200" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen2000" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen400" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen200" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen400" --scene_name "waldo_kitchen"


# python render_lerf_by_text.py -m "./output/kitchen0.1_200" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen0.1_400" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen0.1_800" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen0.1_1000" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen0.1_1200" --scene_name "waldo_kitchen"
# python render_lerf_by_text.py -m "./output/kitchen0.1_200" --scene_name "waldo_kitchen"

# python render_lerf_by_text.py -m "./output/ff3c1a8d-c" --scene_name "teatime"
# python render_lerf_by_text.py -m "./output/2a610f01-5" --scene_name "waldo_kitchen"


python render_lerf_by_text.py -m "./output/0.02_teatime" --scene_name "teatime"
# python render_lerf_by_text.py -m "./output/figurines_sparse_our3e5" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines_sparse_base" --scene_name "figurines"

# python render_lerf_by_text.py -m "./output/figurines0.1_200" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines0.1_1000" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines0.1_1200" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines_cluster_800" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines_cluster_900" --scene_name "figurines"


# python render_lerf_by_text.py -m "./output/ramen1" --scene_name "ramen"
# python render_lerf_by_text.py -m "./output/500" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/600" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/800" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/10" --scene_name "figurines"