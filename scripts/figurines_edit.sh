eval "$(conda shell.bash hook)"
conda activate opengs
export CUDA_VISIBLE_DEVICES=0

# python render_lerf_by_text.py -m "./output/figurines50" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines100" --scene_name "figurines"
python render_lerf_by_text.py -m "./output/0.04" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/0.03" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/0.04" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/0.05" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/0.06" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/0.07" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/0.1" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/0.2" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines100" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines600" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines400" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines1000" --scene_name "figurines"
# python render_lerf_by_text.py -m "./output/figurines1200" --scene_name "figurines"