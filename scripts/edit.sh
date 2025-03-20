source ~/anaconda3/etc/profile.d/conda.sh

# 激活指定的 conda 环境
conda activate opengs
export CUDA_VISIBLE_DEVICES=3
# python render_lerf_by_text.py -m "./output/IMG8304" --scene_name "IMG8304"
python render_lerf_by_text.py -m "./output/40a45240-c" --scene_name "figurines"