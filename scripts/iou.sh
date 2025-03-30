source ~/anaconda3/etc/profile.d/conda.sh

# 激活指定的 conda 环境
conda activate opengs
export CUDA_VISIBLE_DEVICES=1
# python render_lerf_by_text.py -m "./output/IMG8304" --scene_name "IMG8304"
# python render_lerf_by_text.py -m "./output/100" --scene_name "figurines"

# python scripts/compute_lerf_iou.py --scene_name "waldo_kitchen"
# python scripts/compute_lerf_iou.py --scene_name "figurines"
# python scripts/compute_lerf_iou.py --scene_name "teatime"
python scripts/compute_lerf_iou.py --scene_name "ramen"