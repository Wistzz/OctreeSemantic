#!/bin/bash
# chmod +x scripts/train_lerf.sh
# ./scripts/train_lerf.sh

# !!! Please check the dataset path specified by -s.
# 假设你的 Anaconda 安装在默认路径
eval "$(conda shell.bash hook)"
conda activate opengs
###############################################
#              (1/4) figurines
# Training takes approximately 70 minutes on a 24G 4090 GPU.
# The object selection effect is better (recommended), the point cloud visualization is poor (not recommended).
# k1=64, k2=10
# --pos_weight 0.5
# --save_memory: Saves memory, but will reduce training speed. If your GPU memory > 24GB, you can omit this flag
##############################################
scan="figurines"
gpu_num=6 # change
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train_detach_detach.py --port 400$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40000 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 0.1 \
    --temp 0.03 \
    --min_cluster_size 400 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/gaussian-splatting/output/d8bf0888-b/chkpnt30000.pth #/data/sunwei/OctreeSemantic/output/figurines40000/chkpnt40000.pth # #/data/sunwei/OctreeSemantic/output/cea0aa87-8/chkpnt50000.pth #/


# # ###############################################
# # #              (2/4) waldo_kitchen
# # # Training takes approximately 60 minutes on a 24G 4090 GPU.
# # # Good point cloud visualization result (recommended), suboptimal object selection effect.
# # # k1=64, k2=10
# # # --pos_weight 0.5
# # # No need to set save_memory, 24G is sufficient.
# # ###############################################
scan="waldo_kitchen"
gpu_num=6       # change
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train_detach_detach.py --port 441$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40000 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --temp 0.1 \
    --min_cluster_size 1600 \
    --pos_weight 0.5 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/gaussian-splatting/output/waldo_kitchen/chkpnt30000.pth #/data/sunwei/OctreeSemantic/output/kitchen50000/chkpnt50000.pth #

###############################################
#              (3/4) teatime
# Training takes approximately 80 minutes on a 24G 4090 GPU.
# k1=32, k2=10
# --pos_weight 0.1
# --save_memory: Saves memory, but will reduce training speed. If your GPU memory > 24GB, you can omit this flag
###############################################
scan="teatime"
gpu_num=6    # change
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train_detach_detach.py --port 603$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40000 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --temp 0.02 \
    --min_cluster_size 800 \
    --pos_weight 0.5 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/gaussian-splatting/output/teatime/chkpnt30000.pth #/data/sunwei/OctreeSemantic/output/0.02_teatime/chkpnt40000.pth #


# ###############################################
# #              (4/4) ramen
# # Training takes approximately 40 minutes on a 24G 4090 GPU.
# # The object selection effect is the worst and unstable (not recommended).
# # k1=64, k2=10
# # --pos_weight 0.5
# # --loss_weight 0.01: the weight of intra-mask smooth loss. 0.1 is used for the other scenes.
# # No need to set save_memory, 24G is sufficient.
# ###############################################
scan="ramen"
gpu_num=6
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train_detach_detach.py --port 642$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40000 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --temp 0.03 \
    --min_cluster_size 100 \
    --pos_weight 0.5 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/gaussian-splatting/output/ramen/chkpnt30000.pth # /data/sunwei/OctreeSemantic/output/0.03_ramen/chkpnt40000.pth # #

